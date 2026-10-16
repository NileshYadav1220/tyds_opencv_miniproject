# Traffic Sign Classification (Easy)

Uses a small CNN to classify traffic sign images.

## Dataset
Create folders like:
```text
data/train/Stop/
data/train/SpeedLimit/
data/val/Stop/
data/val/SpeedLimit/
```
Put images inside each class folder. Use a GTSRB dataset or a small subset.

## Run
```bash
pip install -r requirements.txt
python app.py
```
The model is saved as `traffic_sign_model.keras`. This version trains a model; it does not include a dataset.
