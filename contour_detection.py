import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to classify shapes based on contour vertices
def classify_shape(contour):
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif num_vertices > 4:
        return "Circle"
    else:
        return "Unknown"

# Function to draw contours and classify shapes
def draw_and_classify_shapes(image, contours):
    image_with_contours = image.copy()
    for contour in contours:
        shape = classify_shape(contour)
        cv2.putText(image_with_contours, shape, (contour[0][0][0], contour[0][0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 3)
    return image_with_contours

# Function to display images
def display_images(image_list, titles):
    plt.figure(figsize=(10, 10))
    for i, (image, title) in enumerate(zip(image_list, titles)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
    plt.tight_layout()
    plt.show()

# Main function to process images
def main():
    # List of images and titles for display
    image_paths = ['images/image1.jpg', 'images/image2.jpg', 'images/image3.jpg', 'images/image4.jpg']
    images = []
    titles = []

    for image_path in image_paths:
        # Load and process each image
        image, gray_image = load_and_convert_image(image_path)
        
        # Apply edge detection techniques
        sobel_edges = sobel_edge_detection(gray_image)
        canny_edges = canny_edge_detection(gray_image)
        log_edges = log_edge_detection(gray_image)
        
        # Find contours using the Canny edge detection result
        contours = find_contours(canny_edges)
        
        # Draw contours and classify shapes
        image_with_contours = draw_and_classify_shapes(image, contours)
        
        # Store the processed images and titles
        images.append(image_with_contours)
        titles.append(f"Contours and Shapes - {image_path.split('/')[-1]}")
    
    # Display all processed images
    display_images(images, titles)

# Run the main function
if __name__ == "__main__":
    main()
