# ===================== TEXT TO SPEECH =====================
import pyttsx3
import time
import asyncio
import edge_tts
import pygame
import tempfile
# =========================================================

# Ultralytics 🚀 AGPL-3.0 License
import argparse
import os
import platform
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

pygame.mixer.init()

async def speak_async(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        filename = f.name

    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)


from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode


# -------- DISTANCE FILTER --------
CLOSE_OBJECT_AREA = 20000
# ---------------------------------


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=0,
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=True,
):

    source = str(source)

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)

    is_speaking = False
    spoken_objects = {}
    disappearance_frames = 20

    save_img = False
    webcam = True

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    model.warmup(imgsz=(1, 3, *imgsz))

    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:

        current_frame_objects = set()

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

            annotator = Annotator(im0, line_width=3, example=str(names))

            if len(det):

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)
                    object_name = names[c]

                    x1, y1, x2, y2 = map(int, xyxy)

                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    # -------- FILTER FAR OBJECTS --------
                    if area < CLOSE_OBJECT_AREA:
                        continue
                    # -----------------------------------

                    frame_width = im0.shape[1]
                    center_x = (x1 + x2) / 2

                    if center_x < frame_width / 3:
                        position = "on your left"
                    elif center_x < 2 * frame_width / 3:
                        position = "in front of you"
                    else:
                        position = "on your right"

                    speech_text = f"{object_name} {position}"

                    current_frame_objects.add(speech_text)

                    if speech_text not in spoken_objects and not is_speaking:

                        print("Speaking:", speech_text)

                        try:
                            is_speaking = True
                            asyncio.run(speak_async(speech_text))
                        except RuntimeError:
                            pass

                        is_speaking = False
                        spoken_objects[speech_text] = 0

                    label = f"{object_name} {position} {conf:.2f}"

                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            for obj in list(spoken_objects.keys()):
                if obj not in current_frame_objects:
                    spoken_objects[obj] += 1
                    if spoken_objects[obj] > disappearance_frames:
                        del spoken_objects[obj]

            if view_img:

                cv2.imshow("Detection", im0)

                if cv2.waitKey(1) == ord("q"):
                    break

        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")


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