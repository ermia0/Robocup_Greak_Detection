import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2 as cv
from ultralytics import YOLO


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Config:
    model_path: str = "/Users/ermia/Desktop/RoboCup2026/yolo/thr_project/runs/detect/train5/weights/best.pt"
    camera_index: int = 0
    frame_w: int = 480
    frame_h: int = 320
    device: str = "mps"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 256
    max_det: int = 1
    v_min: int = 45
    min_area: int = 180
    max_area: int = 90000
    padding_px: int = 12
    padding_ratio: float = 0.10
    inspect_period_sec: float = 0.5
    lock_recheck_sec: float = 2.0
    show_confirm_sec: float = 2.0
    confirm_cooldown_sec: float = 1.0
    lock_min_conf: float = 0.60
    recheck_min_conf: float = 0.62
    recheck_min_match_score: float = 0.05
    read_fail_sleep_sec: float = 0.01
    inset_scale: float = 0.46
    inset_margin: int = 10
    inset_alpha: float = 0.95


@dataclass
class Candidate:
    box: Box
    area: float


@dataclass
class Detection:
    cls_id: int
    conf: float
    name: str
    box: Box


class Mode(str, Enum):
    SEARCH = "SEARCH"
    WAIT_RECHECK = "WAIT_RECHECK"


class VisionPreprocessor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    def normalize(self, frame_bgr):
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    def roi_mask(self, frame_bgr):
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        thr = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 2
        )

        hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
        _, _, v = cv.split(hsv)
        dark = cv.inRange(v, 0, self.cfg.v_min)

        out = cv.bitwise_and(thr, dark)
        out = cv.morphologyEx(out, cv.MORPH_OPEN, self.kernel, iterations=1)
        out = cv.morphologyEx(out, cv.MORPH_CLOSE, self.kernel, iterations=1)
        return out

    def extract_candidates(self, mask) -> List[Candidate]:
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates: List[Candidate] = []
        for c in cnts:
            area = cv.contourArea(c)
            if area < self.cfg.min_area or area > self.cfg.max_area:
                continue
            x, y, w, h = cv.boundingRect(c)
            candidates.append(Candidate((x, y, x + w, y + h), float(area)))
        candidates.sort(key=lambda c: c.area, reverse=True)
        return candidates


class Detector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.model.fuse()

    def predict(self, image_bgr):
        return self.model.predict(
            source=image_bgr,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            device=self.cfg.device,
            verbose=False,
        )[0]

    def best(self, result) -> Optional[Detection]:
        if result.boxes is None or len(result.boxes) == 0:
            return None
        idx = int(result.boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx].tolist())
        cls_id = int(result.boxes.cls[idx].item())
        conf = float(result.boxes.conf[idx].item())
        name = self.model.names.get(cls_id, str(cls_id))
        return Detection(cls_id=cls_id, conf=conf, name=name, box=(x1, y1, x2, y2))


class CandidateMatcher:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def _center(box: Box):
        x1, y1, x2, y2 = box
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def pick(self, frame_shape, candidates: List[Candidate], locked_box: Optional[Box]) -> Optional[Box]:
        if locked_box is None or not candidates:
            return None
        h, w = frame_shape[:2]
        diag = max(1.0, (w * w + h * h) ** 0.5)
        lc_x, lc_y = self._center(locked_box)

        best_box = None
        best_score = -1e18
        for cand in candidates:
            box = cand.box
            ov = self._iou(locked_box, box)
            cx, cy = self._center(box)
            dist = (((cx - lc_x) ** 2 + (cy - lc_y) ** 2) ** 0.5) / diag
            score = (4.0 * ov) - dist
            if score > best_score:
                best_score = score
                best_box = box

        if best_box is None or best_score < self.cfg.recheck_min_match_score:
            return None
        return best_box


class CompetitionController:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pre = VisionPreprocessor(cfg)
        self.detector = Detector(cfg)
        self.matcher = CandidateMatcher(cfg)

        self.mode = Mode.SEARCH
        self.next_inspect_at = 0.0
        self.recheck_at = 0.0
        self.confirmed_until = 0.0
        self.confirm_cooldown_until = 0.0

        self.candidate_idx = 0
        self.status = "boot"
        self.confirmed_name = "NONE"

        self.locked_cls: Optional[int] = None
        self.locked_name = "NONE"
        self.locked_box: Optional[Box] = None

    def _pad_clip(self, box: Box, frame_shape) -> Box:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px = self.cfg.padding_px + int(bw * self.cfg.padding_ratio)
        py = self.cfg.padding_px + int(bh * self.cfg.padding_ratio)
        x1 = max(0, min(x1 - px, w - 1))
        y1 = max(0, min(y1 - py, h - 1))
        x2 = max(0, min(x2 + px, w))
        y2 = max(0, min(y2 + py, h))
        return x1, y1, x2, y2

    def _inspect_candidate(self, frame, box: Box):
        roi = self._pad_clip(box, frame.shape)
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return None, roi, None
        crop = self.pre.normalize(frame[y1:y2, x1:x2])
        if crop.size == 0:
            return None, roi, None
        result = self.detector.predict(crop)
        det = self.detector.best(result)
        return det, roi, crop

    def _reset_lock(self, now: float) -> None:
        self.mode = Mode.SEARCH
        self.locked_cls = None
        self.locked_name = "NONE"
        self.locked_box = None
        self.recheck_at = 0.0
        self.next_inspect_at = now + self.cfg.inspect_period_sec

    def step(self, frame, now: float):
        norm = self.pre.normalize(frame)
        mask = self.pre.roi_mask(frame)
        candidates = self.pre.extract_candidates(mask)

        if candidates:
            self.candidate_idx = min(self.candidate_idx, len(candidates) - 1)
        else:
            self.candidate_idx = 0

        active_box = None
        debug_crop = None

        if now > self.confirmed_until:
            self.confirmed_name = "NONE"

        if self.mode == Mode.SEARCH and now >= self.next_inspect_at and now >= self.confirm_cooldown_until:
            if not candidates:
                self.status = "no contour"
                self.next_inspect_at = now + self.cfg.inspect_period_sec
            else:
                cand = candidates[self.candidate_idx]
                det, roi, crop = self._inspect_candidate(frame, cand.box)
                active_box = roi
                debug_crop = crop

                if det is None:
                    self.status = "miss -> next"
                    self.candidate_idx = (self.candidate_idx + 1) % len(candidates)
                    self.next_inspect_at = now + self.cfg.inspect_period_sec
                elif det.conf < self.cfg.lock_min_conf:
                    self.status = f"low conf {det.conf:.2f} -> next"
                    self.candidate_idx = (self.candidate_idx + 1) % len(candidates)
                    self.next_inspect_at = now + self.cfg.inspect_period_sec
                else:
                    self.locked_cls = det.cls_id
                    self.locked_name = det.name
                    self.locked_box = roi
                    self.recheck_at = now + self.cfg.lock_recheck_sec
                    self.mode = Mode.WAIT_RECHECK
                    self.status = f"hit {det.name}, lock"

        if self.mode == Mode.WAIT_RECHECK:
            if self.locked_box is not None:
                active_box = self.locked_box
            if now >= self.recheck_at:
                recheck_box = self.matcher.pick(frame.shape, candidates, self.locked_box)
                if recheck_box is None:
                    self.status = "lock lost -> next"
                    if candidates:
                        self.candidate_idx = (self.candidate_idx + 1) % len(candidates)
                    self._reset_lock(now)
                else:
                    det2, roi2, crop2 = self._inspect_candidate(frame, recheck_box)
                    active_box = roi2
                    debug_crop = crop2
                    if (
                        det2 is not None
                        and self.locked_cls is not None
                        and det2.cls_id == self.locked_cls
                        and det2.conf >= self.cfg.recheck_min_conf
                    ):
                        self.confirmed_name = self.locked_name
                        self.confirmed_until = now + self.cfg.show_confirm_sec
                        self.confirm_cooldown_until = now + self.cfg.confirm_cooldown_sec
                        self.status = f"confirmed {self.confirmed_name}"
                    else:
                        self.status = "recheck fail -> next"
                        if candidates:
                            self.candidate_idx = (self.candidate_idx + 1) % len(candidates)
                    self._reset_lock(now)

        return {
            "norm": norm,
            "mask": mask,
            "candidates_len": len(candidates),
            "active_box": active_box,
            "debug_crop": debug_crop,
            "status": self.status,
            "confirmed_name": self.confirmed_name,
            "mode": self.mode.value,
            "lock_name": self.locked_name,
            "lock_left": self.recheck_at - now,
        }


class Renderer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def draw_hud(self, out, fps: float, meta) -> None:
        _, w = out.shape[:2]
        cv.rectangle(out, (0, 0), (w, 160), (20, 20, 20), -1)
        cv.putText(out, f"FPS: {fps:.1f}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(
            out,
            f"MODE: {meta['mode']}  CAND: {meta['candidates_len']}",
            (10, 55),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv.putText(out, f"STATUS: {meta['status']}", (10, 85), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(
            out,
            f"CONFIRMED: {meta['confirmed_name']}",
            (10, 115),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if meta["mode"] == Mode.WAIT_RECHECK.value:
            cv.putText(
                out,
                f"LOCKED: {meta['lock_name']}  t={max(0.0, meta['lock_left']):.1f}s",
                (10, 145),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

    def draw_active_box(self, out, active_box: Optional[Box], mode: str) -> None:
        if active_box is None:
            return
        x1, y1, x2, y2 = active_box
        color = (0, 255, 255) if mode == Mode.WAIT_RECHECK.value else (255, 200, 0)
        cv.rectangle(out, (x1, y1), (x2, y2), color, 2)

    def draw_inset(self, out, norm, mask, active_box: Optional[Box], label: str) -> None:
        oh, ow = out.shape[:2]
        ih = max(80, int(oh * self.cfg.inset_scale))
        iw = max(120, int(ow * self.cfg.inset_scale))

        mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        inset_src = cv.addWeighted(norm, 0.8, mask_bgr, 0.2, 0)
        inset = cv.resize(inset_src, (iw, ih), interpolation=cv.INTER_AREA)

        if active_box is not None:
            x1, y1, x2, y2 = active_box
            sx = iw / float(max(1, ow))
            sy = ih / float(max(1, oh))
            rx1 = max(0, min(iw - 1, int(x1 * sx)))
            ry1 = max(0, min(ih - 1, int(y1 * sy)))
            rx2 = max(0, min(iw - 1, int(x2 * sx)))
            ry2 = max(0, min(ih - 1, int(y2 * sy)))
            cv.rectangle(inset, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)

        cv.putText(inset, label[:24], (10, ih - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        x2 = ow - self.cfg.inset_margin
        y1 = self.cfg.inset_margin
        x1 = x2 - iw
        y2 = y1 + ih
        if x1 < 0 or y2 > oh:
            return

        dst = out[y1:y2, x1:x2]
        cv.addWeighted(inset, self.cfg.inset_alpha, dst, 1.0 - self.cfg.inset_alpha, 0.0, dst)
        cv.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), 2)

    def render(self, meta, fps: float):
        out = meta["norm"].copy()
        self.draw_active_box(out, meta["active_box"], meta["mode"])
        self.draw_hud(out, fps, meta)
        inset_label = meta["confirmed_name"] if meta["confirmed_name"] != "NONE" else meta["status"]
        self.draw_inset(out, meta["norm"], meta["mask"], meta["active_box"], inset_label)
        return out


def run(cfg: Config) -> None:
    cap = cv.VideoCapture(cfg.camera_index)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.frame_w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.frame_h)
    try:
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("camera is not captured")
        return

    ctrl = CompetitionController(cfg)
    renderer = Renderer(cfg)

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(cfg.read_fail_sleep_sec)
            continue

        if frame.shape[1] != cfg.frame_w or frame.shape[0] != cfg.frame_h:
            frame = cv.resize(frame, (cfg.frame_w, cfg.frame_h))

        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0
            t0 = now

        meta = ctrl.step(frame, now)
        out = renderer.render(meta, fps)
        cv.imshow("out", out)

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run(Config())
