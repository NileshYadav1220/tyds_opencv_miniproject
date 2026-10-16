# Easy Plant Leaf Disease Classification using transfer learning
import tensorflow as tf
from tensorflow.keras import layers, models

SIZE = (224, 224)
train = tf.keras.utils.image_dataset_from_directory(
    "data/train", image_size=SIZE, batch_size=16
)
val = tf.keras.utils.image_dataset_from_directory(
    "data/val", image_size=SIZE, batch_size=16
)

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(train.class_names), activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=3)
model.save("plant_model.keras")
print("Saved plant_model.keras")
