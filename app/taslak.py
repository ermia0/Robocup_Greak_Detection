import time
import cv2 as cv
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "/Users/ermia/Desktop/RoboCup2026/yolo/thr_project/runs/detect/train5/weights/last.pt"

CONF = 0.50
IOU = 0.5
IMGSZ = 256
MAX_DET = 1
DEVICE = "mps"

V_MIN = 45
MIN_AREA = 300
MAX_AREA = 150000

PADDING_PX = 15
PADDING_RATIO = 0.10

model = YOLO(MODEL_PATH)
model.fuse()

cv.setUseOptimized(True)
cv.setNumThreads(0)

def tone_separate_gray(gray, black_pt=35, white_pt=150, gamma=0.75):
    g = gray.astype(np.float32)
    g = (g - black_pt) / max(1.0, float(white_pt - black_pt))
    g = np.clip(g, 0.0, 1.0)
    g = g ** gamma
    return (g * 255.0).astype(np.uint8)

def clahe_gray(gray):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def only_black_adaptive_from_gray(gray, frame_bgr):
    # gray: CLAHE + tone separated (tek kanal)
    thr = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        31, 2
    )

    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    bm = cv.inRange(v, 0, V_MIN)

    out = cv.bitwise_and(thr, bm)

    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    out = cv.morphologyEx(out, cv.MORPH_OPEN, k, iterations=1)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, k, iterations=1)
    return out

def extract_biggest_box(thr):
    cnts, _ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in cnts:
        area = cv.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if area > best_area:
            x, y, w, h = cv.boundingRect(c)
            best = (x, y, x + w, y + h)
            best_area = area
    return best

def apply_padding(x1, y1, x2, y2):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = PADDING_PX + int(bw * PADDING_RATIO)
    pad_y = PADDING_PX + int(bh * PADDING_RATIO)
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

def clamp(x1, y1, x2, y2, w, h):
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

def draw_best_yolo(r, img_bgr):
    if r.boxes is None or len(r.boxes) == 0:
        cv.putText(img_bgr, "NO DET", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    b = r.boxes
    i = int(b.conf.argmax())
    x1, y1, x2, y2 = map(int, b.xyxy[i].tolist())
    cls_id = int(b.cls[i].item())
    conf = float(b.conf[i].item())
    name = model.names.get(cls_id, str(cls_id))

    cv.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(img_bgr, f"{name} {conf*100:.0f}%",
               (x1, max(y1 - 8, 20)),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def run():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("cap is not captured")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv.resize(frame, (480, 320))
        out = frame.copy()

        # FPS
        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0
            t0 = now

        cv.putText(out, f"FPS: {fps:.1f}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 1) tek sefer gray üret
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = clahe_gray(gray)
        gray = tone_separate_gray(gray)

        # 2) thr mask
        thr = only_black_adaptive_from_gray(gray, frame)
        box = extract_biggest_box(thr)

        if box is not None:
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2)
            x1, y1, x2, y2 = clamp(x1, y1, x2, y2, w, h)

            if x2 > x1 and y2 > y1:
                cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

                crop = frame[y1:y2, x1:x2]  # YOLO'ya renkli crop
                if crop.size:
                    r = yolo(crop)
                    draw_best_yolo(r, crop)
                    cv.imshow("crop", crop)

        cv.imshow("out", out)
        cv.imshow("thr", thr)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run()
