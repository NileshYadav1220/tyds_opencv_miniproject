# Easy Visual Question Answering with pretrained BLIP
from PIL import Image
from transformers import pipeline

path = input("Enter image path: ")
question = input("Ask a question about the image: ")
image = Image.open(path).convert("RGB")

vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
answers = vqa(image, question)

if answers:
    print("Answer:", answers[0]["answer"])
else:
    print("No answer found.")
