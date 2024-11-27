#!/usr/bin/env python3

import os
import cv2
import numpy as np

# Paths to folders
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, "/home/rosmatch/catkin_ws_berisha/src/visual-cable-local-planner/image_processing/images/raw/")
processed_dir = os.path.join(script_dir, "/home/rosmatch/catkin_ws_berisha/src/visual-cable-local-planner/image_processing/images/processed/")
results_dir = os.path.join(script_dir, "/home/rosmatch/catkin_ws_berisha/src/visual-cable-local-planner/image_processing/images/results/")

# Ensure output directories exist
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def process_image(image_path):
    try:
        # Step 1: Read the image from the folder
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print(f"Unable to read image: {image_path}")
            return

        # Step 2: Convert the image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 4: Perform Canny edge detection
        edges = cv2.Canny(blurred, 100, 200)

        # Step 5: Use morphological operations to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Step 6: Find Contours and Filter
        final_result = cv_image.copy()
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5:  # Long and thin shape filtering
                    cv2.rectangle(final_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(final_result, "Cable", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Resize all images to the same dimensions
        target_size = (400, 300)  # Width x Height
        original_resized = cv2.resize(cv_image, target_size)
        edges_resized = cv2.resize(edges, target_size)
        closed_edges_resized = cv2.resize(closed_edges, target_size)
        final_resized = cv2.resize(final_result, target_size)

        # Convert grayscale images to BGR for concatenation
        edges_resized = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
        closed_edges_resized = cv2.cvtColor(closed_edges_resized, cv2.COLOR_GRAY2BGR)

        # Combine all images into a 2x2 grid
        top_row = cv2.hconcat([original_resized, edges_resized])
        bottom_row = cv2.hconcat([closed_edges_resized, final_resized])
        combined_image = cv2.vconcat([top_row, bottom_row])

        # Save the combined image
        combined_output_path = os.path.join(results_dir, f"combined_{os.path.basename(image_path)}")
        cv2.imwrite(combined_output_path, combined_image)

        print(f"Processed and saved results for: {image_path}")
        print(f"Combined image saved at: {combined_output_path}")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def main():
    # Process all images in the raw directory
    image_files = [f for f in os.listdir(raw_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    if not image_files:
        print("No images found in the raw directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(raw_dir, image_file)
        process_image(image_path)

    print("All images processed!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
