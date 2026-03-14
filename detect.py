# ===================== TEXT TO SPEECH =====================
import pyttsx3
import asyncio
import edge_tts
import pygame
import tempfile
import threading
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


# ===================== ASYNC SPEECH =====================

def speak(text):
    """Run speech asynchronously so detection does not freeze"""
    threading.Thread(target=lambda: asyncio.run(speak_async(text)), daemon=True).start()


async def speak_async(text):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        filename = f.name

    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)


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


# ===================== MAIN RUN FUNCTION =====================

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=0,
    imgsz=(320, 320),
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    view_img=True,
):

    source = str(source)

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    spoken_objects = {}
    disappearance_frames = 20
    last_navigation_message = None

    device = select_device(device)

    model = DetectMultiBackend(weights, device=device)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Skip frames to reduce lag on Raspberry Pi
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, vid_stride=2)

    model.warmup(imgsz=(1, 3, *imgsz))

    seen, windows, dt = 0, [], (
        Profile(device=device),
        Profile(device=device),
        Profile(device=device),
    )

    for path, im, im0s, vid_cap, s in dataset:

        current_frame_objects = set()

        left_obstacle = False
        center_obstacle = False
        right_obstacle = False

        # ================= PREPROCESS =================

        with dt[0]:

            im = torch.from_numpy(im).to(model.device)
            im = im.float()
            im /= 255

            if len(im.shape) == 3:
                im = im[None]

        # ================= INFERENCE =================

        with dt[1]:
            pred = model(im)

        # ================= NMS =================

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):

            im0 = im0s[i].copy()

            annotator = Annotator(im0, line_width=3, example=str(names))

            if len(det):

                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape
                ).round()

                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)
                    object_name = names[c]

                    x1, y1, x2, y2 = map(int, xyxy)

                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    # Ignore far objects
                    if area < CLOSE_OBJECT_AREA:
                        continue

                    frame_width = im0.shape[1]
                    center_x = (x1 + x2) / 2

                    # Determine zone
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

                    if speech_text not in spoken_objects:

                        print("Speaking:", speech_text)

                        speak(speech_text)

                        spoken_objects[speech_text] = 0

                    label = f"{object_name} {position} {conf:.2f}"

                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            # ================= SMART NAVIGATION =================

            navigation_message = None

            if not center_obstacle:
                navigation_message = "Path clear ahead"

            elif center_obstacle and not left_obstacle:
                navigation_message = "Obstacle ahead move left"

            elif center_obstacle and not right_obstacle:
                navigation_message = "Obstacle ahead move right"

            else:
                navigation_message = "Obstacle on both sides move carefully"

            if navigation_message != last_navigation_message:

                print("Navigation:", navigation_message)

                speak(navigation_message)

                last_navigation_message = navigation_message

            # ================= CLEAN MEMORY =================

            for obj in list(spoken_objects.keys()):

                if obj not in current_frame_objects:

                    spoken_objects[obj] += 1

                    if spoken_objects[obj] > disappearance_frames:
                        del spoken_objects[obj]

            # ================= DISPLAY WINDOW =================

            if view_img:

                cv2.imshow("Detection", im0)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")
        cv2.destroyAllWindows()


# ===================== CLI =====================

def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default="yolov5s.pt")
    parser.add_argument("--source", type=str, default="0")

    opt = parser.parse_args()

    print_args(vars(opt))

    return opt


def main(opt):

    run(**vars(opt))


if __name__ == "__main__":

    opt = parse_opt()

    main(opt)