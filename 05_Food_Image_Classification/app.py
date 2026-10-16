# Easy image classification using pretrained MobileNetV2
import cv2
import tensorflow as tf

path = input("Enter food image path: ")
image = cv2.imread(path)

if image is None:
    print("Image not found.")
    raise SystemExit

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = tf.image.resize(image, (224, 224))
image = tf.keras.applications.mobilenet_v2.preprocess_input(image.numpy())
model = tf.keras.applications.MobileNetV2(weights="imagenet")
prediction = model.predict(image[None, ...], verbose=0)

for _, label, score in tf.keras.applications.mobilenet_v2.decode_predictions(
        prediction, top=5)[0]:
    print(label, round(float(score) * 100, 2), "%")
