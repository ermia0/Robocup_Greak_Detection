"""import time
import cv2 as cv
from ultralytics import YOLO

# -------- CONFIG --------
MODEL_PATH = "/Users/ermia/Desktop/RoboCup2026/yolo/thr_project/runs/detect/train/weights/last.pt"

CONF = 0.75
IOU = 0.5
IMGSZ = 192          # 256 yerine 192 genelde FPS arttırır
MAX_DET = 1
DEVICE = "mps"

# model.names içinde omega/phi/psi hangi id ise onu yaz:
# örnek: omega=0 phi=1 psi=2 ise:
CLASSES = [0, 1, 2]  # sadece bu class'lara izin ver
# ------------------------

model = YOLO(MODEL_PATH)
model.fuse()

state = {"mono": True}  # toggle bug yok


def mono_pre(img_bgr):
    g = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    g = cv.GaussianBlur(g, (3, 3), 0)
    return cv.cvtColor(g, cv.COLOR_GRAY2BGR)


def yolo(img_bgr):
    # classes filtresi: yanlış şeylere "psi" demeyi azaltır
    return model.predict(
        source=img_bgr,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        classes=CLASSES,
        device=DEVICE,
        verbose=False
    )[0]


def draw_best(r, img_bgr):
    if r.boxes is None or len(r.boxes) == 0:
        cv.putText(img_bgr, "NO DET", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    b = r.boxes
    i = int(b.conf.argmax())

    x1, y1, x2, y2 = map(int, b.xyxy[i].tolist())
    cls_id = int(b.cls[i].item())
    conf = float(b.conf[i].item())

    name = model.names.get(cls_id, str(cls_id))
    label = f"{name} {conf*100:.0f}%"

    cv.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(img_bgr, label, (x1, max(y1 - 8, 20)),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def run():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # bazı kameralar buffer yüzünden gecikme yapar
    try:
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0
            t0 = now

        inp = mono_pre(frame) if state["mono"] else frame
        r = yolo(inp)

        out = frame.copy()
        draw_best(r, out)

        cv.putText(out, f"FPS: {fps:.1f} | mono: {state['mono']}",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)

        cv.imshow("YOLO FULL", out)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('m'):
            state["mono"] = not state["mono"]
            print("mono =", state["mono"])

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run()
"""


import time
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from collections import deque

MODEL_PATH = "/Users/ermia/Desktop/RoboCup2026/yolo/thr_project/runs/detect/train5/weights/best.pt"

CONF = 0.30
IOU = 0.45
IMGSZ = 256
MAX_DET = 5
DEVICE = "mps"

V_MIN = 45
MIN_AREA = 300
MAX_AREA = 150000

PADDING_PX = 12
PADDING_RATIO = 0.10

# doğrulama
STEP1_SEC = 1.0
STEP2_SEC = 2.0
SHOW_CONFIRM_SEC = 2.0

# lock state
cand_q = deque()
locked_cls = None
stage = 0          # 0 idle, 1 wait step1, 2 wait step2
check_t = 0.0

confirmed_cls = None
confirmed_t = 0.0

model = YOLO(MODEL_PATH)
model.fuse()


def normalize_light(frame_bgr):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def only_black_adaptive(norm_bgr, raw_bgr):
    gray = cv.cvtColor(norm_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    thr = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        31, 2
    )

    hsv = cv.cvtColor(raw_bgr, cv.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    bm = cv.inRange(v, 0, V_MIN)

    out = cv.bitwise_and(thr, bm)

    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    out = cv.morphologyEx(out, cv.MORPH_OPEN, k, iterations=1)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, k, iterations=1)
    return out


def extract_boxes(thr):
    cnts, _ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv.boundingRect(c)
        boxes.append((x, y, x + w, y + h, area))
    return boxes


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def pick_best_box(boxes, w, h):
    """
    Basit stabil seçim:
    - merkeze yakın + büyük alan
    """
    if not boxes:
        return None
    cx0, cy0 = w * 0.5, h * 0.5

    best = None
    best_score = -1e18
    for x1, y1, x2, y2, area in boxes:
        cx, cy = box_center((x1, y1, x2, y2))
        dist = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
        score = area - 2.0 * dist   # dist arttıkça ceza
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)
    return best



def pad_and_clamp(box, w, h):
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = PADDING_PX + int(bw * PADDING_RATIO)
    py = PADDING_PX + int(bh * PADDING_RATIO)

    x1 -= px; y1 -= py; x2 += px; y2 += py

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def yolo(img_bgr):
    return model.predict(
        source=img_bgr,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        device=DEVICE,
        verbose=False
    )[0]


def best_cls_id(r):
    if r.boxes is None or len(r.boxes) == 0:
        return None
    i = int(r.boxes.conf.argmax())
    return int(r.boxes.cls[i].item())


def top_unique_classes(r, k=5):
    if r.boxes is None or len(r.boxes) == 0:
        return []

    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy().astype(float)
    idx = np.argsort(-conf)

    uniq = []
    seen = set()
    for i in idx:
        c = int(cls[i])
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
        if len(uniq) >= k:
            break
    return uniq


def draw_best_yolo_box(r, img_bgr):
    if r.boxes is None or len(r.boxes) == 0:
        cv.putText(img_bgr, "NO DET", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return

    i = int(r.boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
    cls_id = int(r.boxes.cls[i].item())
    conf = float(r.boxes.conf[i].item())

    name = model.names.get(cls_id, str(cls_id))
    cv.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv.putText(img_bgr, f"{name} {conf:.2f}",
               (x1, max(y1 - 8, 20)),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def lock_and_verify(now, cur_best, uniq_list):
    """
    İstenen davranış:
    - Bir adayı seç -> lock
    - step1 süresince başka aday arama yok
    - step1 kontrol: tutarsa stage2'ye kilitli geç
    - tutmazsa sıradaki adaya geç
    """
    global cand_q, locked_cls, stage, check_t, confirmed_cls, confirmed_t

    # yeni tur başlat
    if stage == 0:
        cand_q = deque(uniq_list)
        locked_cls = cand_q.popleft() if cand_q else None
        if locked_cls is not None:
            stage = 1
            check_t = now + STEP1_SEC
        return

    # lock yoksa reset
    if locked_cls is None:
        stage = 0
        return

    # zaman gelince kontrol
    if now < check_t:
        return

    # kontrol zamanı
    if cur_best == locked_cls:
        if stage == 1:
            stage = 2
            check_t = now + STEP2_SEC
        else:
            confirmed_cls = locked_cls
            confirmed_t = now
            print("ONAYLANDI:", model.names.get(confirmed_cls, str(confirmed_cls)))

            # reset
            stage = 0
            locked_cls = None
            cand_q.clear()
    else:
        # olmadı → sıradaki adaya geç
        if cand_q:
            locked_cls = cand_q.popleft()
            stage = 1
            check_t = now + STEP1_SEC
        else:
            stage = 0
            locked_cls = None


def run():
    cap = cv.VideoCapture(0)

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv.resize(frame, (480, 320))
        now = time.time()

        # FPS
        frames += 1
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0
            t0 = now

        norm = normalize_light(frame)
        thr = only_black_adaptive(norm, frame)
        boxes = extract_boxes(thr)
        box = pick_best_box(boxes, frame.shape[1], frame.shape[0])



        out = norm.copy()
        cv.putText(out, f"FPS {fps:.1f}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cur_best = None
        uniq_list = []

        if box is not None:
            x1, y1, x2, y2 = pad_and_clamp(box, frame.shape[1], frame.shape[0])
            if x2 > x1 and y2 > y1:
                # ROI box
                cv.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)

                crop = norm[y1:y2, x1:x2].copy()
                if crop.size != 0:
                    r = yolo(crop)
                    cur_best = best_cls_id(r)
                    uniq_list = top_unique_classes(r, k=5)

                    # crop üstünde yolo box
                    draw_best_yolo_box(r, crop)
                    cv.imshow("crop", crop)

                    # lock + verify
                    lock_and_verify(now, cur_best, uniq_list)

        # minimal text (ama informative)
        best_name = model.names.get(cur_best, "NONE") if cur_best is not None else "NONE"
        cv.putText(out, f"BEST: {best_name}", (10, 55),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if locked_cls is not None:
            ln = model.names.get(locked_cls, str(locked_cls))
            cv.putText(out, f"LOCK: {ln}  stage={stage}", (10, 85),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if confirmed_cls is not None and (now - confirmed_t) <= SHOW_CONFIRM_SEC:
            fn = model.names.get(confirmed_cls, str(confirmed_cls))
            cv.putText(out, f"ONAYLANDI: {fn}", (10, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv.imshow("out", out)
        cv.imshow("thr", thr)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run()