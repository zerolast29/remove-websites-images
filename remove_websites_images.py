import cv2
import os
import numpy as np

# Path to input and output folders
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def auto_detect_crop_y(image):
    """Automatically detect where to crop based on content."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute horizontal projection to find empty white space at the bottom
    projection = np.sum(gray, axis=1)
    threshold = np.mean(projection) * 0.9  # Adjust threshold dynamically
    
    # Find first row from bottom where content is detected
    for y in range(len(projection) - 1, 0, -1):
        if projection[y] < threshold:
            return y  # Crop above this line
    return len(projection)  # Default to full height if no clear cutoff

def crop_image(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    # Auto-detect crop height
    Y_END = auto_detect_crop_y(image)
    cropped_image = image[:Y_END, :]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped and saved: {output_path}")

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
        crop_image(input_path, output_path)

if __name__ == "__main__":
    process_images()
    print("Processing complete. Check the output_images folder.")
