# Easy handwritten/printed text OCR
import cv2
import easyocr

image_path = input("Enter image path (example: handwriting.jpg): ")
image = cv2.imread(image_path)

if image is None:
    print("Image not found. Check the path.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(gray, detail=0)
    text = "\n".join(results)
    print("\nRecognized text:\n", text)
    with open("recognized_text.txt", "w", encoding="utf-8") as file:
        file.write(text)
    print("Saved to recognized_text.txt")
