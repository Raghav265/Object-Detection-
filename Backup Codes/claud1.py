# ===================== TEXT TO SPEECH =====================
import pyttsx3
import threading
import queue
import time
# =========================================================

import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# ===================== SPEECH ENGINE =====================

class SpeechEngine:
    """
    Offline TTS using pyttsx3.
    - Windows : uses SAPI5 (built-in, natural voice, no install needed)
    - Mac     : uses 'say' command (built-in, no install needed)
    - Linux   : uses espeak (install: sudo apt install espeak -y)

    Two queues:
      - nav_queue : size 1, navigation messages (highest priority)
      - obj_queue : size 3, object name announcements

    Speech worker always drains nav_queue first so navigation
    warnings can never be dropped by object announcements.

    Cooldown is tracked per object CLASS (e.g. "person"), not per
    full phrase, so position changes always get re-announced.
    """

    SPEAK_COOLDOWN = 2      # seconds between same-class object announcements
    NAV_COOLDOWN   = 2      # seconds between navigation messages

    def __init__(self):
        self.nav_queue = queue.Queue(maxsize=1)
        self.obj_queue = queue.Queue(maxsize=3)

        # Cooldown tracking — keyed by object class name
        self.last_spoken_class = {}   # class_name -> last spoken phrase
        self.last_spoken_time  = {}   # class_name -> timestamp
        self.last_nav_message  = None
        self.last_nav_time     = 0

        self._is_windows = platform.system() == "Windows"

        # On Linux/Pi: reuse a single engine instance (thread-safe there)
        # On Windows : create a fresh engine per utterance to avoid
        #              "run loop already started" threading conflicts
        if not self._is_windows:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 150)
            self._engine.setProperty("volume", 1.0)

        self._worker = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker.start()

    def _say(self, text):
        """Internal: actually speak the text. Platform-aware."""
        print(f"[TTS] {text}")
        try:
            if self._is_windows:
                # Fresh engine per call — avoids Windows COM threading issues
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            else:
                # Reuse shared engine on Linux/Pi
                self._engine.say(text)
                self._engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    def _speech_worker(self):
        """Background thread: drain nav_queue first, then obj_queue."""
        while True:
            # Always check navigation queue first
            try:
                text = self.nav_queue.get(block=False)
                self._say(text)
                continue
            except queue.Empty:
                pass

            # Then handle object announcements
            try:
                text = self.obj_queue.get(timeout=0.1)
                self._say(text)
            except queue.Empty:
                pass

    def speak_object(self, class_name, phrase):
        """
        Announce a detected object.
        Cooldown is per class_name so position changes always get announced.
        """
        now = time.time()
        last_phrase = self.last_spoken_class.get(class_name)
        last_time   = self.last_spoken_time.get(class_name, 0)

        position_changed = (last_phrase != phrase)
        cooldown_passed  = (now - last_time > self.SPEAK_COOLDOWN)

        if position_changed or cooldown_passed:
            self.last_spoken_class[class_name] = phrase
            self.last_spoken_time[class_name]  = now
            try:
                self.obj_queue.put_nowait(phrase)
            except queue.Full:
                # Drop oldest object announcement to make room
                try:
                    self.obj_queue.get_nowait()
                except queue.Empty:
                    pass
                self.obj_queue.put_nowait(phrase)

    def speak_navigation(self, message):
        """
        Announce a navigation message.
        Only spoken when message changes AND nav cooldown has passed.
        nav_queue size=1 so stale nav messages are replaced by latest.
        """
        now = time.time()
        message_changed = (message != self.last_nav_message)
        cooldown_passed = (now - self.last_nav_time > self.NAV_COOLDOWN)

        if message_changed and cooldown_passed:
            self.last_nav_message = message
            self.last_nav_time    = now
            # Replace any pending nav message — latest is always most relevant
            try:
                self.nav_queue.get_nowait()
            except queue.Empty:
                pass
            self.nav_queue.put_nowait(message)

    def speak_now(self, text):
        """
        Speak immediately, bypassing both queues.
        Used for startup, shutdown, and camera error messages.
        """
        self._say(text)


# ===================== YOLO IMPORTS =====================

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


# ===================== NAVIGATION LOGIC =====================

def get_position(center_x, frame_width):
    """Classify object horizontal position into left / center / right."""
    if center_x < frame_width / 3:
        return "on your left"
    elif center_x < 2 * frame_width / 3:
        return "in front of you"
    else:
        return "on your right"


def get_navigation_message(left_obstacle, center_obstacle, right_obstacle):
    """
    Return a navigation instruction string based on obstacle positions.
    Returns None when path is clear — the caller handles the
    transition-only "path clear" announcement.
    """
    if not center_obstacle:
        return None                             # path is clear
    elif not left_obstacle:
        return "Obstacle ahead, move left"
    elif not right_obstacle:
        return "Obstacle ahead, move right"
    else:
        return "Obstacle on both sides, move carefully"


# ===================== MAIN RUN =====================

@smart_inference_mode()
def run(
    weights="yolov5n.pt",
    source="0",
    imgsz=(160, 160),       # 160x160 for faster inference on Pi (was 256x256)
    conf_thres=0.25,
    iou_thres=0.45,
    min_area=20000,          # ignore objects smaller than this (px²)
    device="",
    display=False,           # headless by default; pass --display for debugging
):
    tts = SpeechEngine()
    tts.speak_now("System started")

    device = select_device(device)
    model  = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz  = check_img_size(imgsz, s=stride)

    # Tracks previous frame's path state for transition-only "path clear"
    path_was_clear = True

    # Outer loop handles camera disconnection and reconnection
    while True:
        try:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, vid_stride=2)
            model.warmup(imgsz=(1, 3, *imgsz))

            dt = (
                Profile(device=device),
                Profile(device=device),
                Profile(device=device),
            )

            for path, im, im0s, vid_cap, s in dataset:

                left_obstacle   = False
                center_obstacle = False
                right_obstacle  = False

                # ---- preprocess ----
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.float() / 255
                    if len(im.shape) == 3:
                        im = im[None]

                # ---- inference ----
                with dt[1]:
                    pred = model(im)

                # ---- NMS ----
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres)

                # ---- process detections ----
                for i, det in enumerate(pred):
                    im0       = im0s[i].copy()
                    annotator = Annotator(im0, line_width=3, example=str(names))

                    if len(det):
                        det[:, :4] = scale_boxes(
                            im.shape[2:], det[:, :4], im0.shape
                        ).round()

                        for *xyxy, conf, cls in reversed(det):
                            c          = int(cls)
                            class_name = names[c]
                            x1, y1, x2, y2 = map(int, xyxy)
                            area       = (x2 - x1) * (y2 - y1)

                            # Skip objects that are too small / too far
                            if area < min_area:
                                continue

                            frame_width = im0.shape[1]
                            center_x    = (x1 + x2) / 2
                            position    = get_position(center_x, frame_width)

                            # Track which zones have obstacles
                            if position == "on your left":
                                left_obstacle = True
                            elif position == "in front of you":
                                center_obstacle = True
                            else:
                                right_obstacle = True

                            phrase = f"{class_name} {position}"
                            tts.speak_object(class_name, phrase)

                            # Draw box only when --display is active
                            if display:
                                annotator.box_label(
                                    xyxy,
                                    f"{class_name} {position} {conf:.2f}",
                                    color=colors(c, True),
                                )

                    # ---- navigation (transition-only "path clear") ----
                    nav_msg = get_navigation_message(
                        left_obstacle, center_obstacle, right_obstacle
                    )

                    if nav_msg is None:
                        # Path is clear this frame
                        if not path_was_clear:
                            # Was blocked last frame — announce transition once
                            tts.speak_navigation("Path clear")
                        path_was_clear = True
                    else:
                        tts.speak_navigation(nav_msg)
                        path_was_clear = False

                    # ---- display window (debug only) ----
                    if display:
                        im0 = annotator.result()
                        cv2.imshow("Detection", im0)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            cv2.destroyAllWindows()
                            tts.speak_now("System stopped")
                            return

                LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")

        except KeyboardInterrupt:
            # Let KeyboardInterrupt bubble up to main()
            raise

        except Exception as e:
            print(f"[ERROR] {e}")
            tts.speak_now("Camera error, restarting")
            time.sleep(3)       # brief pause before reconnect attempt
            continue            # restart the camera loop


# ===================== CLI =====================

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",    type=str,   default="yolov5n.pt",
        help="Model weights file"
    )
    parser.add_argument(
        "--source",     type=str,   default="0",
        help="Camera index or video file path"
    )
    parser.add_argument(
        "--imgsz",      type=int,   nargs="+", default=[160, 160],
        help="Inference image size as two ints: height width"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-thres",  type=float, default=0.45,
        help="NMS IoU threshold"
    )
    parser.add_argument(
        "--min-area",   type=int,   default=20000,
        help="Minimum bounding box area in px² to announce an object"
    )
    parser.add_argument(
        "--device",     type=str,   default="",
        help="Device: cpu, 0, 1, etc."
    )
    parser.add_argument(
        "--display",    action="store_true",
        help="Show annotated video window (for debugging only)"
    )
    opt = parser.parse_args()

    # Convert imgsz list to tuple for run()
    opt.imgsz = tuple(opt.imgsz)

    print_args(vars(opt))
    return opt


def main(opt):
    try:
        run(**vars(opt))
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
        # Speak shutdown message directly — queues may already be dead
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say("System stopped")
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)