import cv2
import numpy as np
import os

# Function to read and convert image to grayscale
def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image
# Function to apply Sobel edge detection
def sobel_edge_detection(gray_image):
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y)
# Function to apply Canny edge detection
def canny_edge_detection(gray_image, low_threshold=100, high_threshold=200):
    return cv2.Canny(gray_image, low_threshold, high_threshold)
# Function to apply Laplacian of Gaussian (LoG) edge detection
def log_edge_detection(gray_image):
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return cv2.Laplacian(blurred_image, cv2.CV_64F)
# Function to find contours from an edge-detected image
def find_contours(edge_image):
    # Convert the edge image to an 8-bit format if it's not already
    edge_image = cv2.convertScaleAbs(edge_image)  # Convert to 8-bit
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
def auto_label_shape(contour):
    # Approximate the contour to a polygon
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)  # 4% tolerance
    vertices_count = len(approx)

    # Calculate the contour area and perimeter
    area = cv2.contourArea(contour)

    # Special cases for very small contours
    if area < 10 and vertices_count < 4:
        return 'Unknown'  # Too small to determine

    # Circle detection: if the contour is round and has a high number of vertices
    if vertices_count > 5:
        # Use aspect ratio or perimeter/area ratio to determine if it's a circle
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.8:  # Circularity threshold (ideal circle is close to 1)
            return 'Circle'

    # Triangle detection
    if vertices_count == 3:
        return 'Triangle'

    # Square/Rectangle detection (4 vertices)
    if vertices_count == 4:
        # Check if the shape is approximately square
        rect = cv2.boundingRect(approx)
        width, height = rect[2], rect[3]
        aspect_ratio = width / height
        if 0.9 <= aspect_ratio <= 1.1:  # Threshold for square
            return 'Square'
        else:
            return 'Rectangle'

    # Handle other polygons
    if vertices_count > 4:
        return 'Polygon'

    # Default case for unidentified shapes
    return 'Unknown'

# Inside your contour processing loop, replace the manual labeling with the auto_label_shape function:
def calculate_shape_features(image, contours):
    features = []
    labels = []  # Automatically generated labels based on the shape

    for contour in contours:
        # Calculate the contour features (area, perimeter, etc.)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        vertices = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
        
        # Pixel count: Calculate the number of pixels in the contour mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        pixel_count = cv2.countNonZero(mask)
        
        # Create a dictionary for features
        feature_dict = {
            "area": area,
            "perimeter": perimeter,
            "vertices": vertices,
            "pixel_count": pixel_count
        }
        
        features.append(feature_dict)
        print(feature_dict)
        # Auto label the shape
        labels.append(auto_label_shape(contour))  # Use the automated labeling function

    return features, labels
# Function to draw contours and display calculated features
def draw_and_display_features(image, contours, features):
    image_with_contours = image.copy()
    
    # Aggregate statistics
    total_area = sum(feature["area"] for feature in features)
    total_perimeter = sum(feature["perimeter"] for feature in features)
    total_vertices = sum(feature["vertices"] for feature in features)
    total_pixel_count = sum(feature["pixel_count"] for feature in features)
    
    # Display summary statistics on the image
    cv2.putText(image_with_contours, f"Total Area: {total_area:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image_with_contours, f"Total Perim: {total_perimeter:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image_with_contours, f"Total Verts: {total_vertices}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image_with_contours, f"Total Pixels: {total_pixel_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw the contours and add labels
    for i, contour in enumerate(contours):
        # Get the label for the shape based on contours
        label = auto_label_shape(contour)
        
        # Get the centroid of the contour to position the label
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Ensure the contour is not empty
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw the contour
        cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)
        
        # Add the label near the centroid of the contour
        cv2.putText(image_with_contours, label, (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image_with_contours
# Function to save the image with the new label
def save_image_with_label(image, original_image_path):
    # Extract the original image name without the extension
    image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    
    # Create the "kNN guessed" image filename
    output_filename = f"{image_name}_kNN_guessed.jpg"
    output_path = os.path.join('final_images', output_filename)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")
# Main function to process images and save to "final_images" folder
def main():
    # List of images for processing
    image_paths = ['images/simple_circle.jpeg', 'images/simple_square.webp', 'images/simple_triangle.jpg', 'images/simple_yin_and_yang.jpg']
    
    # Create folder "final_images" if it doesn't exist
    output_folder = 'final_images'
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        # Load and process each image
        image, gray_image = load_and_convert_image(image_path)
        
        # Apply edge detection techniques (you can experiment with different methods)
        sobel_edges = sobel_edge_detection(gray_image)
        canny_edges = canny_edge_detection(gray_image)
        log_edges = log_edge_detection(gray_image)
        
        # Find contours for each edge detection method
        sobel_contours = find_contours(sobel_edges)
        canny_contours = find_contours(canny_edges)
        log_contours = find_contours(log_edges)
        
        # Calculate features (area, perimeter, vertices count, pixel count) for each set of contours
        sobel_features, _ = calculate_shape_features(image, sobel_contours)
        canny_features, _ = calculate_shape_features(image, canny_contours)
        log_features, _ = calculate_shape_features(image, log_contours)

        # Draw contours and display features
        sobel_image = draw_and_display_features(image, sobel_contours, sobel_features)
        canny_image = draw_and_display_features(image, canny_contours, canny_features)
        log_image = draw_and_display_features(image, log_contours, log_features)
        
        # Save the processed images to the "final_images" folder
        sobel_output_path = os.path.join(output_folder, f"sobel_{os.path.basename(image_path)}")
        canny_output_path = os.path.join(output_folder, f"canny_{os.path.basename(image_path)}")
        log_output_path = os.path.join(output_folder, f"log_{os.path.basename(image_path)}")
        
        cv2.imwrite(sobel_output_path, sobel_image)
        cv2.imwrite(canny_output_path, canny_image)
        cv2.imwrite(log_output_path, log_image)
        
        print(f"Saved: {sobel_output_path}")
        print(f"Saved: {canny_output_path}")
        print(f"Saved: {log_output_path}")

# Run the main function
if __name__ == "__main__":
    main()
