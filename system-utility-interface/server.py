from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUNS = sorted((PROJECT / "artifacts").glob("federated-*"), key=lambda p: p.stat().st_mtime)
if not RUNS:
    raise RuntimeError("No federated model found. Run train.py before starting the interface.")
RUN = RUNS[-1]
MODEL = torch.jit.load(str(RUN / "model_scripted.pt"), map_location="cpu").eval()
CLASS_TO_IDX = json.loads((RUN / "classes.json").read_text(encoding="utf-8"))
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
PIPELINE = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="Privacy-Preserving FL MobileViT")
app.mount("/app", StaticFiles(directory=HERE / "app"), name="app")


@app.get("/")
def home():
    return FileResponse(HERE / "index.html")


@app.get("/api/status")
def status():
    return {"ready": True, "model": RUN.name, "classes": list(CLASS_TO_IDX)}


def prepare(raw: bytes):
    try:
        image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise HTTPException(400, "The uploaded file is not a readable image.") from exc
    gray = np.array(image)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    tensor = PIPELINE(Image.fromarray(rgb)).unsqueeze(0)
    return rgb, tensor


def saliency_data_url(rgb: np.ndarray, tensor: torch.Tensor, class_index: int) -> str:
    x = tensor.detach().requires_grad_(True)
    MODEL.zero_grad()
    MODEL(x)[0, class_index].backward()
    saliency = x.grad.detach().abs().amax(dim=1)[0].numpy()
    saliency -= saliency.min()
    saliency /= max(float(saliency.max()), 1e-8)
    heat = cv2.applyColorMap(np.uint8(saliency * 255), cv2.COLORMAP_JET)
    base = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (256, 256))
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds the 20 MB limit.")
    rgb, tensor = prepare(raw)
    with torch.inference_mode():
        probabilities = MODEL(tensor).softmax(1)[0]
    predicted = int(probabilities.argmax())
    return {
        "prediction": IDX_TO_CLASS[predicted],
        "confidence": float(probabilities[predicted]),
        "probabilities": {IDX_TO_CLASS[i]: float(value) for i, value in enumerate(probabilities)},
        "attention_map": saliency_data_url(rgb, tensor, predicted),
        "model": RUN.name,
        "disclaimer": "Research decision-support output; not a clinical diagnosis.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

