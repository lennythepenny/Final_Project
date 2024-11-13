import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Function to read and convert image to grayscale
def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

# Function to apply Sobel edge detection
def sobel_edge_detection(gray_image):
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(magnitude)  # Convert to 8-bit image

# Function to apply Canny edge detection
def canny_edge_detection(gray_image, low_threshold=100, high_threshold=200):
    return cv2.Canny(gray_image, low_threshold, high_threshold)

# Function to apply Laplacian of Gaussian (LoG) edge detection
def log_edge_detection(gray_image):
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    log_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
    return cv2.convertScaleAbs(log_edges)  # Convert to 8-bit image

# Function to find contours from an edge-detected image
def find_contours(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to draw contours on the image
def draw_contours_on_image(image, contours):
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)  # Green contours
    return image_with_contours

# Sample labeled training data for shapes (area, perimeter, vertices count, pixel count, label)
training_data = [
    [500.0, 90.0, 4, 500, 'Square'],
    [600.0, 75.0, 3, 600, 'Triangle'],
    [700.0, 110.0, 0, 700, 'Circle'],
    [1200.0, 180.0, 4, 1200, 'Rectangle'],
    [800.0, 150.0, 8, 800, 'Star'],
    # Add more samples as needed for training
]

# Prepare training features and labels
X_train = [data[:4] for data in training_data]  # Use area, perimeter, vertices, pixel_count
y_train = [data[4] for data in training_data]   # Labels

# Train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Function to classify contours based on k-NN model
def classify_contour_with_knn(image, contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    vertices = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
    mask = create_mask(image, contour)  # Create mask for pixel count
    pixel_count = count_pixels_in_contour(mask)  # Count pixels inside contour
    contour_features = [area, perimeter, vertices, pixel_count]
    shape = knn.predict([contour_features])[0]
    return shape

# Create a binary mask for a contour
def create_mask(image, contour):
    # Create a black image of the same size as the original image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Fill the contour area with white (255) on the mask
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    return mask

# Function to count pixels inside a contour
def count_pixels_in_contour(mask):
    return cv2.countNonZero(mask)  # Count non-zero pixels in the mask

# Function to draw contours and classify shapes using k-NN
def draw_and_classify_shapes(image, contours):
    image_with_contours = image.copy()
    for contour in contours:
        shape = classify_contour_with_knn(image, contour)
        cv2.putText(image_with_contours, shape, (contour[0][0][0], contour[0][0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 3)
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
        
        # Apply traditional edge detection methods
        sobel_edges = sobel_edge_detection(gray_image)
        canny_edges = canny_edge_detection(gray_image)
        log_edges = log_edge_detection(gray_image)
        
        # Find contours using each edge detection result
        sobel_contours = find_contours(sobel_edges)
        canny_contours = find_contours(canny_edges)
        log_contours = find_contours(log_edges)
        
        # Draw and save the contours for traditional methods
        sobel_image = draw_contours_on_image(image, sobel_contours)
        canny_image = draw_contours_on_image(image, canny_contours)
        log_image = draw_contours_on_image(image, log_contours)

        # Draw and classify contours using KNN
        knn_image = draw_and_classify_shapes(image, canny_contours)  # Using Canny for KNN
        
        # Save the processed images to the "final_images" folder
        sobel_output_path = os.path.join(output_folder, f"sobel_{os.path.basename(image_path)}")
        canny_output_path = os.path.join(output_folder, f"canny_{os.path.basename(image_path)}")
        log_output_path = os.path.join(output_folder, f"log_{os.path.basename(image_path)}")
        knn_output_path = os.path.join(output_folder, f"knn_{os.path.basename(image_path)}")
        
        cv2.imwrite(sobel_output_path, sobel_image)
        cv2.imwrite(canny_output_path, canny_image)
        cv2.imwrite(log_output_path, log_image)
        cv2.imwrite(knn_output_path, knn_image)
        
        print(f"Saved: {sobel_output_path}, {canny_output_path}, {log_output_path}, {knn_output_path}")

# Run the main function
if __name__ == "__main__":
    main()
