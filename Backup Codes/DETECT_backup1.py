# ===================== TEXT TO SPEECH =====================
# =========================================================
# Ultralytics 🚀 AGPL-3.0 License
import argparse
import asyncio
import os
import platform
import sys
import tempfile
from pathlib import Path

import edge_tts
import pygame
import pyttsx3
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
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):

    source = str(source)

    # -------- VOICE SETUP --------
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)

    is_speaking = False
    spoken_objects = {}
    disappearance_frames = 20
    # -----------------------------

    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")

    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        current_frame_objects = set()
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred = model(im, augment=augment, visualize=False)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, _frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, _frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    object_name = names[c]

                    # Get frame width
                    frame_width = im0.shape[1]

                    # Get object center position
                    center_x = (xyxy[0] + xyxy[2]) / 2

                    # Determine the psoition
                    if center_x < frame_width / 3:
                        position = "on your left"
                    elif center_x < 2 * frame_width / 3:
                        position = "in front of you"
                    else:
                        position = "on your right"

                    # Create full speech message
                    speech_text = f"{object_name} {position}"

                    current_frame_objects.add(speech_text)

                    # -------- SPEAK ONLY ONCE PER OBJECT --------
                    if speech_text not in spoken_objects and not is_speaking:
                        print("Speaking:", speech_text)
                        try:
                            is_speaking = True
                            asyncio.run(speak_async(speech_text))
                        except RuntimeError:
                            pass
                        is_speaking = False
                        spoken_objects[speech_text] = 0
                    # --------------------------------------------

                    label = None if hide_labels else f"{object_name} {position} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()
            # Update spoken memory after frame
            for obj in list(spoken_objects.keys()):
                if obj not in current_frame_objects:
                    spoken_objects[obj] += 1
                    if spoken_objects[obj] > disappearance_frames:
                        del spoken_objects[obj]

            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")

    if update:
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default=ROOT / "data/images")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
