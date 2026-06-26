import time
import threading
import cv2
import torch
import numpy as np
import torchvision.ops as ops
import supervision as sv

from model import YOLOv3
from config import DEVICE, NUM_CLASSES, ANCHORS

# =========================
# SETTINGS
# =========================
RTMP_URL = "rtmp://172.20.10.5:1935/live/stream"
CHECKPOINT_PATH = "checkpoint.pth.tar"
INPUT_SIZE = 416
CONF_THRESH = 0.35
IOU_THRESH = 0.35
USE_FP16 = torch.cuda.is_available()
RECORD_OUTPUT = True

CLASS_NAMES = [
    "pedestrian",
    "person",
    "bicycle",
    "car",
    "van",
    "truck",
    "bus",
    "motorcycle",
]

CLASS_CONF_THRESHOLDS = {
    0: 0.65,  # pedestrian
    4: 0.60,  # van
    7: 0.60,  # motorcycle
}

CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 0, 255),
    2: (255, 255, 0),
    3: (0, 255, 0),
    4: (255, 0, 0),
    5: (0, 0, 255),
    6: (255, 255, 255),
    7: (128, 0, 128),
}


# =========================
# HELPER FUNCTIONS
# =========================
def letterbox_image(frame, new_size=416, color=(128, 128, 128)):
    h, w = frame.shape[:2]
    scale = new_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)

    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def unletterbox_box(x, y, w, h, scale, pad_x, pad_y, orig_w, orig_h, input_size):
    x = x * input_size
    y = y * input_size
    w = w * input_size
    h = h * input_size

    x1 = (x - w / 2 - pad_x) / scale
    y1 = (y - h / 2 - pad_y) / scale
    x2 = (x + w / 2 - pad_x) / scale
    y2 = (y + h / 2 - pad_y) / scale

    # Expand the box a little
    box_w = x2 - x1
    box_h = y2 - y1

    expand_w = box_w * 0.08
    expand_h = box_h * 0.12

    x1 -= expand_w
    x2 += expand_w
    y1 -= expand_h
    y2 += expand_h

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(orig_w - 1, x2))
    y2 = int(min(orig_h - 1, y2))

    return x1, y1, x2, y2


def preds_to_sv_detections(preds, scale, pad_x, pad_y, orig_w, orig_h, input_size):
    xyxy = []
    confidences = []
    class_ids = []

    for p in preds:
        x, y, w, h, score, cls_id = p
        cls_id = int(cls_id)
        score = float(score)

        # Use special threshold for certain classes, otherwise default threshold
        class_thresh = CLASS_CONF_THRESHOLDS.get(cls_id, CONF_THRESH)
        if score < class_thresh:
            continue

        x1, y1, x2, y2 = unletterbox_box(
            x, y, w, h, scale, pad_x, pad_y, orig_w, orig_h, input_size
        )

        if x2 > x1 and y2 > y1:
            xyxy.append([x1, y1, x2, y2])
            confidences.append(score)
            class_ids.append(cls_id)

        if not xyxy:
            return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=np.int32),
    )


def draw_tracked_predictions(frame, tracked_detections):
    if tracked_detections.xyxy is None or len(tracked_detections.xyxy) == 0:
        return frame

    for i in range(len(tracked_detections.xyxy)):
        x1, y1, x2, y2 = tracked_detections.xyxy[i].astype(int)

        cls_id = int(tracked_detections.class_id[i])
        conf = float(tracked_detections.confidence[i])

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - th - 10)
        label_y2 = max(0, y1)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw, label_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return frame


# =========================
# VIDEO CAPTURE CLASS
# =========================
class LatestFrameCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.running = True

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


# =========================
# YOLO PREDICTION DECODING
# =========================
def get_nms_predictions(outputs, anchors, conf_thresh, iou_thresh):
    all_boxes = []
    all_scores = []
    all_classes = []

    for scale_idx, out in enumerate(outputs):
        S = out.shape[2]
        anchors_scale = anchors[scale_idx].to(DEVICE)

        obj_probs = torch.sigmoid(out[0, ..., 4])
        mask = obj_probs > conf_thresh
        if not mask.any():
            continue

        valid_out = out[0][mask]
        class_probs = torch.softmax(valid_out[..., 5:], dim=-1)
        max_class_conf, max_class_idx = torch.max(class_probs, dim=-1)
        scores = obj_probs[mask] * max_class_conf

        score_mask = scores > conf_thresh
        if not score_mask.any():
            continue

        valid_out = valid_out[score_mask]
        scores = scores[score_mask]
        max_class_idx = max_class_idx[score_mask]

        indices = mask.nonzero(as_tuple=False)[score_mask]
        a_idx = indices[:, 0]
        j = indices[:, 1]
        i = indices[:, 2]

        tx = valid_out[..., 0]
        ty = valid_out[..., 1]
        tw = valid_out[..., 2]
        th = valid_out[..., 3]

        bx = (torch.sigmoid(tx) + i) / S
        by = (torch.sigmoid(ty) + j) / S
        bw = anchors_scale[a_idx, 0] * torch.exp(tw.clamp(-10, 10))
        bh = anchors_scale[a_idx, 1] * torch.exp(th.clamp(-10, 10))

        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(max_class_idx)

    if not all_boxes:
        return []

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    classes = torch.cat(all_classes, dim=0)

    keep = ops.batched_nms(boxes, scores, classes, iou_thresh)

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    return torch.stack(
        [cx, cy, w, h, scores, classes.float()],
        dim=1
    ).tolist()


# =========================
# MAIN
# =========================
def main():
    model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False)
    model.eval()

    if USE_FP16:
        model.half()

    cap = LatestFrameCapture(RTMP_URL)
    tracker = sv.ByteTrack()

    video_out = None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h_orig, w_orig = frame.shape[:2]

        if RECORD_OUTPUT and video_out is None:
            video_out = cv2.VideoWriter(
                "drone_record.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                12.0,
                (w_orig, h_orig),
            )

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb, scale, px, py = letterbox_image(img_rgb, new_size=INPUT_SIZE)

        img_t = torch.from_numpy(lb).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float() / 255.0
        if USE_FP16:
            img_t = img_t.half()

        with torch.no_grad():
            outputs = model(img_t)
            preds = get_nms_predictions(outputs, ANCHORS, CONF_THRESH, IOU_THRESH)

        # 1. Convert model output to detections object
        detections = preds_to_sv_detections(preds, scale, px, py, w_orig, h_orig, INPUT_SIZE)

        # 2. KILL DOUBLE BOXES (PyTorch NMS)
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            keep_idx = ops.nms(
                torch.from_numpy(detections.xyxy).float(),
                torch.from_numpy(detections.confidence).float(),
                iou_threshold=0.45,
            )

            keep_idx = keep_idx.cpu().numpy()
            detections = sv.Detections(
                xyxy=detections.xyxy[keep_idx],
                confidence=detections.confidence[keep_idx],
                class_id=detections.class_id[keep_idx],
            )

        # 3. Update tracker with the cleaned detections
        tracked_detections = tracker.update_with_detections(detections)

        # 4. Draw the results
        display_frame = draw_tracked_predictions(frame.copy(), tracked_detections)

        fps = 1.0 / max(time.time() - prev_time, 1e-6)
        prev_time = time.time()

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if video_out is not None:
            video_out.write(display_frame)

        cv2.imshow("Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if video_out is not None:
        video_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()