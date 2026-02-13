import time
import cv2 as cv
from ultralytics import YOLO
import numpy as np


MODEL_PATH = "/Users/ermia/Desktop/RoboCup2026/yolo/thr_project/runs/detect/train5/weights/last.pt"

COSTUME_CONF=75
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

def mono_pre(img_bgr):
    g = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
#    g = cv.GaussianBlur(g, (3, 3), 0)
    return cv.cvtColor(g, cv.COLOR_GRAY2BGR)
def tone_separate(gray, black_pt=35, white_pt=150, gamma=0.75):
    g = gray.astype("float32")
    g = (g - black_pt) / max(1.0, float(white_pt - black_pt))
    g = g.clip(0.0, 1.0)
    g = g ** gamma
    return (g * 255.0).astype("uint8")


def only_black_adaptive(frame_bgr):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    gray = tone_separate(gray, black_pt=35, white_pt=150, gamma=0.75)
#    gray = cv.equalizeHist(gray)
 #   gray = cv.GaussianBlur(gray, (5, 5), 0)

    thr = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        31, 2
    )

    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    _, _, v = cv.split(hsv)
    bm = cv.inRange(v, 0, V_MIN)

    out = cv.bitwise_and(thr, bm)

    k = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    out = cv.morphologyEx(out, cv.MORPH_OPEN, k, iterations=2)
    k = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, k, iterations=2)

    return out

def normalize_light(frame_bgr):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    
#    gray = cv.GaussianBlur(gray, (3, 3), 0)

    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


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
     

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue
        frame = cv.resize(frame,(480,320))



        # FPS
        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0
            t0 = now

        norm = normalize_light(frame)

        thr = only_black_adaptive(norm)
        box = extract_biggest_box(thr)

        out = norm.copy()   

       
        cv.putText(out, f"FPS: {fps:.1f}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if box is not None:
            x1, y1, x2, y2 = box
            fh, fw = frame.shape[:2]

            x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2)
            x1, y1, x2, y2 = clamp(x1, y1, x2, y2, fw, fh)

            if x2 > x1 and y2 > y1:
                cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

                norm = frame
                crop = tone_separate(normalize_light(norm[y1:y2, x1:x2]))

                if crop.size != 0:
                    inp = normalize_light(crop)   
                    r = yolo(inp)
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
