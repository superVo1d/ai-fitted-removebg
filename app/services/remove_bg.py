import base64 as b64
from io import BytesIO

import requests
from PIL import Image
from rembg import remove


def remove_bg_of_img(image_url: str | None = None, image_base64: str | None = None):
    if image_url:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    elif image_base64:
        img = Image.open(BytesIO(b64.b64decode(image_base64)))
    else:
        raise ValueError("Either image_url or image_base64 must be provided")
    output = remove(img)
    out_stream = BytesIO()
    output.save(out_stream, format="PNG")
    return b64.b64encode(out_stream.getvalue()).decode("utf-8")
