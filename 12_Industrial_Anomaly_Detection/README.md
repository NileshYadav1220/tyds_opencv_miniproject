# Industrial Anomaly Detection (Easy)

Learns the appearance of normal images using Isolation Forest, then flags an unusual image.

## Prepare data
Put at least five normal/good product images in one folder.

## Run
```bash
pip install -r requirements.txt
python app.py
```
Enter the normal-image folder, then the image to check. This simple pixel-based model is sensitive to lighting and camera angle; it is a learning demonstration.
