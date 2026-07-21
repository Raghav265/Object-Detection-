# ===================== TEXT TO SPEECH =====================
# =========================================================
import argparse
import asyncio
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue

import cv2
import edge_tts
import pygame
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

pygame.mixer.init()

# ===================== SPEECH CONTROL =====================

speech_queue = Queue(maxsize=3)
last_spoken_time = {}

SPEAK_COOLDOWN = 2
NAV_COOLDOWN = 2


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


def speak(text):
    current_time = time.time()

    if text in last_spoken_time:
        if current_time - last_spoken_time[text] < SPEAK_COOLDOWN:
            return

    last_spoken_time[text] = current_time

    if speech_queue.full():
        try:
            speech_queue.get_nowait()
        except:
            pass

    print("Speaking:", text)
    speech_queue.put(text)


# ===================== BLOCKING SPEECH =====================


def speak_blocking(text):
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

CLOSE_OBJECT_AREA = 20000


# ===================== MAIN RUN =====================


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5n.pt",
    source=0,
    imgsz=(224, 224),  # ✅ reduced for faster startup
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    view_img=True,
):

    source = str(source)

    print("Opening camera...")

    last_navigation_message = None
    last_navigation_time = 0
    last_detected_objects = set()

    system_started = False

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)

    stride, names, _pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # ✅ FAST CAMERA LOAD (IMPORTANT FIX)
    dataset = LoadStreams(
        source,
        img_size=imgsz,
        stride=stride,
        vid_stride=2,
    )

    print("Camera initialized")

    model.warmup(imgsz=(1, 3, *imgsz))

    _seen, _windows, dt = (
        0,
        [],
        (
            Profile(device=device),
            Profile(device=device),
            Profile(device=device),
        ),
    )

    for path, im, im0s, vid_cap, s in dataset:
        # ✅ START SOUND ONLY WHEN FRAME ARRIVES
        if not system_started:
            speak("System started")
            system_started = True

        current_frame_objects = set()

        left_obstacle = False
        center_obstacle = False
        right_obstacle = False

        # PREPROCESS
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float()
            im /= 255

            if len(im.shape) == 3:
                im = im[None]

        # INFERENCE
        with dt[1]:
            pred = model(im)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=3, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    object_name = names[c]

                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)

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
                    current_frame_objects.add(speech_text)

                    if speech_text not in last_detected_objects:
                        speak(speech_text)

                    annotator.box_label(
                        xyxy,
                        f"{object_name} {position} {conf:.2f}",
                        color=colors(c, True),
                    )

            im0 = annotator.result()

            # NAVIGATION
            navigation_message = None

            if not center_obstacle:
                navigation_message = "Path clear ahead"
            elif center_obstacle and not left_obstacle:
                navigation_message = "Obstacle ahead move left"
            elif center_obstacle and not right_obstacle:
                navigation_message = "Obstacle ahead move right"
            else:
                navigation_message = "Obstacle on both sides move carefully"

            current_time = time.time()

            if navigation_message != last_navigation_message and current_time - last_navigation_time > NAV_COOLDOWN:
                speak(navigation_message)
                last_navigation_message = navigation_message
                last_navigation_time = current_time

            last_detected_objects = current_frame_objects.copy()

            # DISPLAY
            if view_img:
                cv2.imshow("Detection", im0)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return

        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")


# ===================== CLI =====================


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5n.pt")
    parser.add_argument("--source", type=str, default="0")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    try:
        run(**vars(opt))
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        speak_blocking("System stopped")  # ✅ STOP SOUND


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
