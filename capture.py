import torch
from PIL import Image
from torchvision import models, transforms

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

# ตั้งค่าระบบ quantization
torch.backends.quantized.engine = "qnnpack"


def load_model(path, device="cpu"):
    model = models.quantization.mobilenet_v3_large(pretrained=True)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(num_ftrs, len(class_dict))
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model


def predict_image(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")  # แปลงภาพให้เป็น RGB
    image = transform(image).unsqueeze(0)  # แปลงภาพและเพิ่ม batch dimension

    with torch.no_grad():
        model = model.to("cpu")
        output = model(image)

    return output


# โหลดภาพ
image_path = "cat.jpg"
model_int8 = load_model("mobilenet_v3_large_CIFA_quantization.pth")
# ตั้งค่าการ preprocess ภาพ
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

output = predict_image(image_path, model_int8, transform)
top = list(enumerate(output[0].softmax(dim=0)))
top.sort(key=lambda x: x[1], reverse=True)
for idx, val in top[:10]:
    print(idx)
    print(f"{val.item()*100:.2f}% {class_dict[idx]}")
