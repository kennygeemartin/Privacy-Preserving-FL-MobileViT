# Privacy-preserving federated MobileViT for Nigerian chest X-rays

This repository downloads the public Kaggle dataset, creates reproducible
70/15/15 stratified splits, partitions the training set across five simulated
hospitals with a Dirichlet distribution, trains MobileViT-XS using FedAvg, and
exports an inference-ready model plus evaluation results.

## Quick start

```powershell
python -m pip install -r requirements.txt
python train.py download
python train.py inspect
python train.py train --mode federated
```

The full paper configuration is in `config.yaml`. For a quick end-to-end check:

```powershell
python train.py train --mode federated --rounds 1 --local-epochs 1 --max-images 80 --no-pretrained
```

Other experiment modes are `centralized` and `local`. Results are written under
`artifacts/<run-name>/`, including `best_model.pt`, `model_scripted.pt`,
`metrics.json`, `classification_report.csv`, `confusion_matrix.csv`, split
manifests, client partitions, and training history.

> This is research software, not a medical device. Predictions require clinical
> validation and must not be used as a substitute for qualified diagnosis.

## System utility interface

The dashboard and model-serving application are in `system-utility-interface`.
It reproduces the five-hospital FL workflow and connects uploaded X-rays to the
latest exported MobileViT model. See `system-utility-interface/README.md` for
launch instructions.

