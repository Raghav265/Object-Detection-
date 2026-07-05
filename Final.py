# ===================== TEXT TO SPEECH =====================
import asyncio
import edge_tts
import pygame
import tempfile
import threading
from queue import Queue, Empty
import time
# =========================================================

import argparse
import os
import sys
from pathlib import Path
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

pygame.mixer.init()

# ===================== SPEECH CONTROL =====================

speech_queue = Queue(maxsize=3)

# Per-object cooldown (time-based, not set-based)
last_spoken_time = {}
SPEAK_COOLDOWN = 3       # seconds between same object announcement
NAV_COOLDOWN = 2         # seconds between navigation messages
PRUNE_INTERVAL = 30      # seconds between pruning last_spoken_time dict
last_prune_time = time.time()


async def speak_async(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            filename = f.name

        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)

        # Clean up temp file after playing
        try:
            os.unlink(filename)
        except Exception:
            pass

    except Exception as e:
        print("TTS Error:", e)


def speech_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        text = speech_queue.get()
        if text is None:
            break
        loop.run_until_complete(speak_async(text))


threading.Thread(target=speech_worker, daemon=True).start()


def prune_last_spoken():
    """Remove old entries from last_spoken_time to prevent memory leak."""
    global last_prune_time
    current_time = time.time()
    if current_time - last_prune_time > PRUNE_INTERVAL:
        cutoff = current_time - SPEAK_COOLDOWN * 10
        keys_to_delete = [k for k, v in last_spoken_time.items() if v < cutoff]
        for k in keys_to_delete:
            del last_spoken_time[k]
        last_prune_time = current_time


def speak(text):
    current_time = time.time()

    # Cooldown check per unique speech text
    if text in last_spoken_time:
        if current_time - last_spoken_time[text] < SPEAK_COOLDOWN:
            return

    last_spoken_time[text] = current_time

    # Drop oldest item if queue is full to keep speech fresh
    if speech_queue.full():
        try:
            speech_queue.get_nowait()
        except Empty:
            pass

    print("Speaking:", text)
    speech_queue.put(text)


def speak_blocking(text):
    """Blocking speech — used only for shutdown message."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(speak_async(text))


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


# ===================== DISTANCE FILTER =====================

CLOSE_OBJECT_AREA = 20000   # Ignore objects smaller than this (pixels²)


# ===================== MAIN RUN =====================

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5n.pt",
    source="0",
    imgsz=(320, 320),       # Reduced from 256 for better accuracy, still fast on Pi
    conf_thres=0.30,        # Slightly higher threshold reduces false positives on Pi
    iou_thres=0.45,
    device="cpu",           # Raspberry Pi has no CUDA GPU
    view_img=True,
):
    last_navigation_message = None
    last_navigation_time = 0

    system_started = False

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # vid_stride=3 skips frames to reduce CPU load on Pi
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, vid_stride=3)

    model.warmup(imgsz=(1, 3, *imgsz))

    seen, windows, dt = 0, [], (
        Profile(device=device),
        Profile(device=device),
        Profile(device=device),
    )

    for path, im, im0s, vid_cap, s in dataset:

        # Announce system start on first frame only
        if not system_started:
            speak("System started")
            system_started = True

        # Prune memory periodically
        prune_last_spoken()

        left_obstacle = False
        center_obstacle = False
        right_obstacle = False

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred = model(im)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):

            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=2, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)
                    object_name = names[c]

                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)

                    # Skip distant/small objects
                    if area < CLOSE_OBJECT_AREA:
                        continue

                    frame_width = im0.shape[1]
                    center_x = (x1 + x2) / 2

                    if center_x < frame_width / 3:
                        position = "on your left"
                        left_obstacle = True
                    elif center_x < 2 * frame_width / 3:
                        position = "in front of you"
                        center_obstacle = True
                    else:
                        position = "on your right"
                        right_obstacle = True

                    speech_text = f"{object_name} {position}"
                    speak(speech_text)   # speak() handles cooldown internally

                    annotator.box_label(
                        xyxy,
                        f"{object_name} {position} {conf:.2f}",
                        color=colors(c, True),
                    )

            im0 = annotator.result()

            # ---- Navigation guidance ----
            current_time = time.time()

            if not center_obstacle:
                navigation_message = "Path clear ahead"
            elif not left_obstacle and not right_obstacle:
                navigation_message = "Obstacle ahead, move left or right"
            elif not left_obstacle:
                navigation_message = "Obstacle ahead, move left"
            elif not right_obstacle:
                navigation_message = "Obstacle ahead, move right"
            else:
                navigation_message = "Obstacles on all sides, move carefully"

            if (
                navigation_message != last_navigation_message
                or current_time - last_navigation_time > NAV_COOLDOWN * 5
            ) and current_time - last_navigation_time > NAV_COOLDOWN:
                speak(navigation_message)
                last_navigation_message = navigation_message
                last_navigation_time = current_time

            if view_img:
                cv2.imshow("Detection", im0)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return

        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")


# ===================== CLI =====================

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5n.pt",
                        help="Path to model weights")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index or video path")
    parser.add_argument("--no-view", action="store_true",
                        help="Disable display window (useful for headless Pi)")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # Allow --no-view flag to disable window on headless Pi
    view_img = not opt.no_view
    try:
        run(weights=opt.weights, source=opt.source, view_img=view_img)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Clear queue then speak stop message
        while not speech_queue.empty():
            try:
                speech_queue.get_nowait()
            except Empty:
                break
        speak_blocking("System stopped")
        pygame.mixer.quit()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)