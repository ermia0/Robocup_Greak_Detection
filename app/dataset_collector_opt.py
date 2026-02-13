import os
import time
import threading
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

import cv2 as cv


@dataclass
class Cfg:
    dataset_root: str = "dataset"

    images_train_dir: str = "images/train"
    labels_train_dir: str = "labels/train"
    images_val_dir: str = "images/val"
    labels_val_dir: str = "labels/val"

    classes: tuple = ("omega", "phi", "psi")
    cap_index: int = 0

    # detect.py ile birebir
    resize_w: int = 480
    resize_h: int = 320

    save_raw_too: bool = True
    save_norm_too: bool = True
    save_crops: bool = True
    crops_dir: str = "crops"

    val_ratio: float = 0.15

    min_area: int = 300
    max_area: int = 150000

    padding_px: int = 15
    padding_ratio: float = 0.10

    v_min: int = 45

    show_windows: bool = True


CFG = Cfg()
CLASS_ID = {name: idx for idx, name in enumerate(CFG.classes)}


def clamp_ratio(v):
    return max(0.01, min(float(v), 0.99))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO dataset collector")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=CFG.val_ratio,
        help="Validation ratio between 0.01 and 0.99 (default: 0.15)",
    )
    return parser.parse_args()


def ensure_dirs():
    img_tr = os.path.join(CFG.dataset_root, CFG.images_train_dir)
    lab_tr = os.path.join(CFG.dataset_root, CFG.labels_train_dir)
    img_va = os.path.join(CFG.dataset_root, CFG.images_val_dir)
    lab_va = os.path.join(CFG.dataset_root, CFG.labels_val_dir)

    os.makedirs(img_tr, exist_ok=True)
    os.makedirs(lab_tr, exist_ok=True)
    os.makedirs(img_va, exist_ok=True)
    os.makedirs(lab_va, exist_ok=True)

    if CFG.save_crops:
        for c in CFG.classes:
            os.makedirs(os.path.join(CFG.dataset_root, CFG.crops_dir, c), exist_ok=True)

    return img_tr, lab_tr, img_va, lab_va


def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def apply_padding(x1, y1, x2, y2):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = CFG.padding_px + int(bw * CFG.padding_ratio)
    pad_y = CFG.padding_px + int(bh * CFG.padding_ratio)

    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y


def yolo_label_line(class_id, x1, y1, x2, y2, img_w, img_h):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n"


# detect.py ile birebir
def tone_separate(gray, black_pt=35, white_pt=150, gamma=0.75):
    g = gray.astype("float32")
    g = (g - black_pt) / max(1.0, float(white_pt - black_pt))
    g = g.clip(0.0, 1.0)
    g = g ** gamma
    return (g * 255.0).astype("uint8")


def normalize_light(frame_bgr):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def only_black_adaptive(frame_bgr):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    gray = tone_separate(gray, black_pt=35, white_pt=150, gamma=0.75)

    thr = cv.adaptiveThreshold(
        gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        31,
        2,
    )

    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    _, _, v = cv.split(hsv)
    bm = cv.inRange(v, 0, CFG.v_min)

    out = cv.bitwise_and(thr, bm)

    k = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    out = cv.morphologyEx(out, cv.MORPH_OPEN, k, iterations=2)
    k = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, k, iterations=2)

    return out


def extract_biggest_box(thr):
    cnts, _ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in cnts:
        area = cv.contourArea(c)
        if area < CFG.min_area or area > CFG.max_area:
            continue
        if area > best_area:
            x, y, w, h = cv.boundingRect(c)
            best = (x, y, x + w, y + h)
            best_area = area

    return best


class FrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame

    def get(self):
        with self.lock:
            if self.latest is None:
                return None
            return self.latest.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


def run():
    img_tr, lab_tr, img_va, lab_va = ensure_dirs()

    cap = cv.VideoCapture(CFG.cap_index)
    if not cap.isOpened():
        print("Kamera acilamadi.")
        return

    grabber = FrameGrabber(cap).start()

    selected = None
    selected_name = "NONE"

    saved_total = 0
    save_count = defaultdict(int)
    split_count = defaultdict(int)

    last_saved_files = []
    last_saved_cls = None
    last_saved_split = None

    t0 = time.time()
    frames = 0
    fps = 0.0

    print("KEYS:")
    print("  1=omega  2=phi  3=psi  0=none")
    print("  S veya SPACE = kaydet")
    print("  U = undo (son kaydi sil)")
    print("  [ / ] = val ratio azalt / arttir")
    print("  ESC = cik")

    while True:
        raw0 = grabber.get()
        if raw0 is None:
            time.sleep(0.01)
            continue

        frame = cv.resize(raw0, (CFG.resize_w, CFG.resize_h))

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
        save_box = None

        cv.putText(out, f"FPS: {fps:.1f}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(out, f"Label: {selected_name} | saved: {saved_total}", (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(out, f"val_ratio: {CFG.val_ratio:.2f}", (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        y = 100
        for name in CFG.classes:
            cv.putText(
                out,
                f"{name}: {save_count[name]} (tr:{split_count[f'{name}_train']} va:{split_count[f'{name}_val']})",
                (10, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            y += 20

        if box is not None:
            x1, y1, x2, y2 = box
            fh, fw = frame.shape[:2]

            x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, fw, fh)

            if x2 > x1 and y2 > y1:
                save_box = (x1, y1, x2, y2)
                cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

                crop = tone_separate(cv.cvtColor(normalize_light(frame[y1:y2, x1:x2]), cv.COLOR_BGR2GRAY))
                crop = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
                if crop.size != 0 and CFG.show_windows:
                    cv.imshow("crop", crop)

        if CFG.show_windows:
            cv.imshow("out", out)
            cv.imshow("thr", thr)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

        if key == ord("0"):
            selected = None
            selected_name = "NONE"
        elif key == ord("1"):
            selected = 0
            selected_name = "omega"
        elif key == ord("2"):
            selected = 1
            selected_name = "phi"
        elif key == ord("3"):
            selected = 2
            selected_name = "psi"

        elif key in (ord("u"), ord("U")):
            if last_saved_files:
                for p in last_saved_files:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                last_saved_files = []
                saved_total = max(0, saved_total - 1)
                if last_saved_cls is not None:
                    save_count[last_saved_cls] = max(0, save_count[last_saved_cls] - 1)
                if last_saved_split is not None and last_saved_cls is not None:
                    key_split = f"{last_saved_cls}_{last_saved_split}"
                    split_count[key_split] = max(0, split_count[key_split] - 1)
                print("UNDO: son kayit silindi")
        elif key == ord("["):
            CFG.val_ratio = clamp_ratio(CFG.val_ratio - 0.01)
            print(f"val_ratio -> {CFG.val_ratio:.2f}")
        elif key == ord("]"):
            CFG.val_ratio = clamp_ratio(CFG.val_ratio + 0.01)
            print(f"val_ratio -> {CFG.val_ratio:.2f}")

        elif key in (ord("s"), ord("S"), 32):
            if selected is None:
                print("Once class sec (1/2/3).")
                continue
            if save_box is None:
                print("Box yok. Harfi kadraja al.")
                continue

            cls_name = CFG.classes[selected]
            class_id = CLASS_ID[cls_name]

            if random.random() < CFG.val_ratio:
                img_dir, lab_dir = img_va, lab_va
                split_name = "val"
            else:
                img_dir, lab_dir = img_tr, lab_tr
                split_name = "train"

            base = f"{cls_name}_{ts()}_cam{CFG.cap_index}"
            x1, y1, x2, y2 = save_box
            label_line = yolo_label_line(class_id, x1, y1, x2, y2, frame.shape[1], frame.shape[0])

            saved_paths = []

            if CFG.save_norm_too:
                img_norm_path = os.path.join(img_dir, base + "_n.jpg")
                txt_norm_path = os.path.join(lab_dir, base + "_n.txt")
                cv.imwrite(img_norm_path, norm)
                with open(txt_norm_path, "w", encoding="utf-8") as f:
                    f.write(label_line)
                saved_paths += [img_norm_path, txt_norm_path]

            if CFG.save_raw_too:
                img_raw_path = os.path.join(img_dir, base + "_r.jpg")
                txt_raw_path = os.path.join(lab_dir, base + "_r.txt")
                cv.imwrite(img_raw_path, frame)
                with open(txt_raw_path, "w", encoding="utf-8") as f:
                    f.write(label_line)
                saved_paths += [img_raw_path, txt_raw_path]

            if CFG.save_crops:
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size != 0:
                    cpath = os.path.join(CFG.dataset_root, CFG.crops_dir, cls_name, f"{base}_crop.png")
                    cv.imwrite(cpath, crop)
                    saved_paths.append(cpath)

            saved_total += 1
            save_count[cls_name] += 1
            split_count[f"{cls_name}_{split_name}"] += 1

            last_saved_files = saved_paths
            last_saved_cls = cls_name
            last_saved_split = split_name

            print(f"SAVED: {base} -> {split_name} | files: {len(saved_paths)}")

    grabber.stop()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    CFG.val_ratio = clamp_ratio(args.val_ratio)
    print(f"Baslangic val_ratio: {CFG.val_ratio:.2f}")
    run()
