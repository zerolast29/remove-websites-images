import cv2
import os
import numpy as np
import pytesseract
from pytesseract import Output

# Path to input and output folders
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Manually set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def remove_website_text(image_path, output_path):
    """Detect and remove website URLs from an image while preserving other content."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract OCR to extract text and bounding boxes
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    
    # Define keywords that commonly appear in website URLs
    website_indicators = [".com", ".org", ".net", "www.", "http", "https"]
    
    for i, text in enumerate(data["text"]):
        if any(indicator in text.lower() for indicator in website_indicators):
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            
            # Draw a white rectangle over the detected text
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)
    
    # Save the modified image
    cv2.imwrite(output_path, image)
    print(f"Processed and saved: {output_path}")

def process_images():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return
    
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".png"))]
    
    if not image_files:
        print("Error: No images found in the input folder.")
        return
    
    for filename in image_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        remove_website_text(input_path, output_path)

if __name__ == "__main__":
    process_images()
    print("Processing complete. Check the output_images folder.")
