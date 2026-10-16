# Plant Disease Classification (Easy)

Uses pretrained MobileNetV2, then trains a small output layer for your leaf classes.

## Dataset folders
```text
data/train/Healthy/
data/train/Diseased/
data/val/Healthy/
data/val/Diseased/
```
Use real class names from a PlantVillage dataset. Add leaf images to each folder.

## Run
```bash
pip install -r requirements.txt
python app.py
```
First run may download MobileNetV2 weights. This is an educational prediction tool, not a confirmed agricultural diagnosis.
