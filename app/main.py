from fastapi import FastAPI
from pydantic import BaseModel
from services.remove_bg import remove_bg_of_img

app = FastAPI()

class Item(BaseModel):
    url: str | None = None
    base64: str | None = None

@app.post("/api/v1/removebg")
def fetch(item: Item):
    try:
        no_bg = remove_bg_of_img(image_url=item.url, image_base64=item.base64)
        return {"image": no_bg}
    except Exception as e:
        return {"content": f"{e}", "status": 500}

@app.get("/api/v1/status")
def status():
    return {"status": "RUNNING"}

