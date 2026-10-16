# Easy image captioning using pretrained BLIP
from PIL import Image
from transformers import pipeline

path = input("Enter image path: ")
image = Image.open(path).convert("RGB")

captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
result = captioner(image, max_new_tokens=35)
caption = result[0]["generated_text"]

print("Image caption:", caption)
with open("caption.txt", "w", encoding="utf-8") as file:
    file.write(caption)
