import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import numpy as np
import os

# Path to input and output folders
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def remove_text_from_image(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to detect text
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    mask = np.zeros_like(gray, dtype=np.uint8)
    
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text.startswith("www.") or text.endswith(".com") or "http" in text:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # Draw mask
    
    # Inpaint to remove the text and preserve background
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # Save the processed image
    cv2.imwrite(output_path, inpainted_image)

def process_images():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            remove_text_from_image(input_path, output_path)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    process_images()
    print("Processing complete. Check the output_images folder.")
