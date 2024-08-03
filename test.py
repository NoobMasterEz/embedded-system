import time

import torch
import numpy as np
from torchvision import models, transforms
import cv2
from PIL import Image

torch.backends.quantized.engine = "qnnpack"
class_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
torch.set_num_threads(2)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)


def load_model(path, device="cpu"):
    model = models.quantization.mobilenet_v3_large(pretrained=True)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(num_ftrs, len(class_dict))
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model


def predict_image(image, model, transform):
    model.eval()
    image = transform(image).unsqueeze(0)  # แปลงภาพและเพิ่ม batch dimension

    with torch.no_grad():
        model = model.to("cpu")
        output = model(image)

    return output


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model_int8 = load_model("mobilenet_v3_large_CIFA_quantization.pth")
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(model_int8)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # convert opencv output from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # run model
        output = predict_image(image, net, preprocess)
        # do something with output ...
        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        for idx, val in top[:10]:
            print(idx)
            print(f"{val.item()*100:.2f}% {class_dict[idx]}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # frame_count += 1
        # now = time.time()
        # if now - last_logged > 1:
        #     print(f"{frame_count / (now-last_logged):.2f} fps")
        #     last_logged = now
        #     frame_count = 0
