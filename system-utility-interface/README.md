# Privacy-Preserving FL MobileViT System Utility

This interface demonstrates the proposed five-hospital federated workflow and
uses the latest exported federated TorchScript model for chest X-ray inference.

## Start the application

From this folder:

```powershell
python -m pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8000`. Select **Initialize Training** to follow the
federated workflow and unlock the diagnostic panel. Upload a JPG or PNG chest
X-ray and select **Run Diagnostics**. The server applies the same CLAHE and
ImageNet normalization pipeline as training, returns four-class probabilities,
and generates an input-gradient saliency overlay.

The server automatically loads the newest `artifacts/federated-*` model from the
parent project. Model binaries and the Kaggle dataset are intentionally excluded
from source control. Run the training pipeline after cloning, and use a completed
full-dataset artifact before reporting or deployment.

This software is a research prototype, not a medical device. Its output must not
replace review by qualified clinical personnel.

