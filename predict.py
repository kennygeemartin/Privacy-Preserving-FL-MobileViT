"""Run inference with an exported research checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms


def preprocess(path: str, image_size: int) -> torch.Tensor:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    image = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return pipeline(image).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model_scripted.pt")
    parser.add_argument("classes", help="Path to classes.json")
    parser.add_argument("image", help="Chest X-ray image")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    model = torch.jit.load(args.model, map_location="cpu").eval()
    class_to_idx = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    with torch.inference_mode():
        probability = model(preprocess(args.image, args.image_size)).softmax(1)[0]
    ranked = sorted(((idx_to_class[i], float(p)) for i, p in enumerate(probability)),
                    key=lambda item: item[1], reverse=True)
    print(json.dumps({"prediction": ranked[0][0], "confidence": ranked[0][1],
                      "probabilities": dict(ranked)}, indent=2))


if __name__ == "__main__":
    main()

