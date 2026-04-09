import base64 as b64
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

_MODEL_ID = os.environ.get("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")

_device: torch.device | None = None
_model: torch.nn.Module | None = None

_transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _device_torch() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        torch.set_float32_matmul_precision("high")
        m = AutoModelForImageSegmentation.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        )
        m.to(_device_torch())
        m.eval()
        _model = m
    return _model


def _remove_bg_pil(image: Image.Image) -> Image.Image:
    im = image.convert("RGB")
    size = im.size
    device = _device_torch()
    model = _load_model()
    batch = _transform_image(im).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(batch)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(size)
    im.putalpha(mask)
    return im


def remove_bg_of_img(image_url: str | None = None, image_base64: str | None = None):
    if image_url:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    elif image_base64:
        img = Image.open(BytesIO(b64.b64decode(image_base64)))
    else:
        raise ValueError("Either image_url or image_base64 must be provided")
    output = _remove_bg_pil(img)
    out_stream = BytesIO()
    output.save(out_stream, format="PNG")
    return b64.b64encode(out_stream.getvalue()).decode("utf-8")
