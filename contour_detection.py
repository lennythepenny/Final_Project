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
    # Approximate the contour to a polygon (with more vertices, the better)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)  # 4% tolerance
    
    # Based on the number of vertices, label the shape
    vertices_count = len(approx)
    
    if vertices_count == 3:
        return 'Triangle'
    elif vertices_count == 4:
        # Check if the width and height are roughly equal (for square/rectangle)
        rect = cv2.boundingRect(approx)
        width, height = rect[2], rect[3]
        if abs(width - height) < 0.1 * width:  # Threshold for "square" (aspect ratio check)
            return 'Square'
        else:
            return 'Rectangle'
    elif vertices_count > 4:
        return 'Polygon'  # or other categories based on vertices count
    else:
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
        
        # Extract features
        features.append([area, perimeter, vertices, pixel_count])
        
        # Auto label the shape
        labels.append(auto_label_shape(contour))  # Use the automated labeling function

    return features, labels

# Function to draw contours and display calculated features
def draw_and_display_features(image, contours, features):
    image_with_contours = image.copy()
    
    for contour, feature in zip(contours, features):
        # Extract feature values
        area, perimeter, vertices, pixel_count = feature
        
        # Display feature values on the image
        cv2.putText(image_with_contours, f"Area: {area:.2f}", (contour[0][0][0], contour[0][0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image_with_contours, f"Perim: {perimeter:.2f}", (contour[0][0][0], contour[0][0][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image_with_contours, f"Verts: {vertices}", (contour[0][0][0], contour[0][0][1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image_with_contours, f"Pixels: {pixel_count}", (contour[0][0][0], contour[0][0][1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw the contours
        cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)
    
    return image_with_contours

# Main function to process images and save to "final_images" folder
def main():
    # List of images for processing
    image_paths = ['images/simple_circle.jpeg', 'images/simple_square.webp', 'images/simple_triangle.jpg', 'images/simple_yin_and_yang.png']
    
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
        sobel_features = calculate_shape_features(image, sobel_contours)
        canny_features = calculate_shape_features(image, canny_contours)
        log_features = calculate_shape_features(image, log_contours)
        
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
