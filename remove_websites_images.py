import cv2
import os
import numpy as np
import pytesseract
import re
from pytesseract import Output

# Path to input and output folders
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Manually set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image, method=1):
    """Apply different preprocessing methods to improve OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 1:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif method == 2:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif method == 3:
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        return cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def is_potential_url(text):
    """Determine if a text segment is a website URL or an excessively long word."""
    website_indicators = [
        ".com", ".org", ".net", "www.", "http", "https", ".edu", ".gov", ".io", ".co",
        ".biz", ".info", ".us", ".uk", ".ca", ".au", "mailto:", "ftp:", "://", ".html", ".php"
    ]
    url_pattern = re.compile(r'https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov|io|co|biz|info|us|uk|ca|au)\S*')

    return (any(indicator in text.lower() for indicator in website_indicators) or 
            url_pattern.search(text) or 
            (len(text) >= 29 and ' ' not in text))  # Detect excessively long words

def detect_and_remove_urls(image):
    """Perform multiple OCR passes and remove detected website text."""
    url_boxes = []

    # Run OCR multiple times with different preprocessing
    for method in [1, 2, 3]:
        processed = preprocess_image(image, method)
        data = pytesseract.image_to_data(processed, output_type=Output.DICT)
        
        for i, text in enumerate(data["text"]):
            if text.strip() and is_potential_url(text):
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                url_boxes.append((x, y, w, h))
    
    # Expand bounding boxes and delete website text
    for x, y, w, h in url_boxes:
        pad_x, pad_y = 80, 25  # Increased padding
        cv2.rectangle(image, (max(0, x - pad_x), max(0, y - pad_y)), 
                      (x + w + pad_x, y + h + pad_y), (255, 255, 255), thickness=-1)

    return image

def remove_website_text(image_path, output_path):
    """Load an image, detect website text, and remove it."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    # Detect and remove URLs
    cleaned_image = detect_and_remove_urls(image)

    # Save the modified image
    cv2.imwrite(output_path, cleaned_image)
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
