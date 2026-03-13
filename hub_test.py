import torch

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Test image
img = "https://ultralytics.com/images/zidane.jpg"

# Inference
results = model(img)

# Output
results.print()
results.show()
results.save()
