# Shape Detection and Edge Detection

This project demonstrates how to apply various edge detection techniques (Sobel, Canny, and Laplacian of Gaussian) to identify and label shapes (circle, triangle, square, rectangle, polygon) in images. The code also calculates various features of the detected shapes (area, perimeter, vertices, and pixel count) and saves the processed images with the shapes' labels.

## Features

- **Edge Detection**: Applies Sobel, Canny, and Laplacian of Gaussian edge detection methods.
- **Shape Detection**: Automatically labels shapes such as circles, triangles, squares, rectangles, and polygons based on contour properties.
- **Shape Features Calculation**: Computes area, perimeter, vertices count, and pixel count for each detected shape.
- **Image Processing**: Loads images, applies edge detection, finds contours, and saves the processed images with labels.
- **Automatic Shape Labeling**: Shapes are labeled based on their contours (e.g., Circle, Triangle, Rectangle, etc.).
- **Save Processed Images**: Saves the resulting images with the detected shapes in the `final_images` folder.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

You can install the required dependencies by running:

## Functions

### **`load_and_convert_image(image_path)`**
Reads an image from the specified path and converts it to grayscale.

- **Parameters**:
  - `image_path` (str): The path to the image.
- **Returns**:
  - The original image and the grayscale image.

---

### **`sobel_edge_detection(gray_image)`**
Applies Sobel edge detection to the grayscale image.

- **Parameters**:
  - `gray_image` (ndarray): The grayscale image.
- **Returns**:
  - The Sobel edge-detected image.

---

### **`canny_edge_detection(gray_image, low_threshold=100, high_threshold=200)`**
Applies Canny edge detection to the grayscale image.

- **Parameters**:
  - `gray_image` (ndarray): The grayscale image.
  - `low_threshold` (int): The lower threshold for Canny.
  - `high_threshold` (int): The upper threshold for Canny.
- **Returns**:
  - The Canny edge-detected image.

---

### **`log_edge_detection(gray_image)`**
Applies Laplacian of Gaussian (LoG) edge detection to the grayscale image.

- **Parameters**:
  - `gray_image` (ndarray): The grayscale image.
- **Returns**:
  - The LoG edge-detected image.

---

### **`find_contours(edge_image)`**
Finds contours in the edge-detected image.

- **Parameters**:
  - `edge_image` (ndarray): The edge-detected image.
- **Returns**:
  - The detected contours.

---

### **`auto_label_shape(contour)`**
Automatically labels a shape based on its contour.

- **Parameters**:
  - `contour` (ndarray): A contour representing a shape.
- **Returns**:
  - The label of the shape (e.g., "Circle", "Triangle", "Rectangle").

---

### **`calculate_shape_features(image, contours)`**
Calculates features (area, perimeter, vertices, pixel count) for each detected contour.

- **Parameters**:
  - `image` (ndarray): The original image.
  - `contours` (list): A list of detected contours.
- **Returns**:
  - A list of feature dictionaries and their corresponding labels.

---

### **`draw_and_display_features(image, contours, features)`**
Draws the contours and displays feature statistics (area, perimeter, vertices, pixel count) on the image.

- **Parameters**:
  - `image` (ndarray): The original image.
  - `contours` (list): A list of detected contours.
  - `features` (list): A list of feature dictionaries.
- **Returns**:
  - The image with drawn contours and feature statistics.

---

### **`save_image_with_label(image, original_image_path)`**
Saves the image with contours and labels to the `final_images` folder.

- **Parameters**:
  - `image` (ndarray): The image with contours and labels.
  - `original_image_path` (str): The path to the original image.

---

### **`main()`**
The main function that processes a list of images, applies edge detection, finds contours, calculates shape features, and saves the processed images.

---

## Usage

1. Place the images you want to process in the `images` folder.
2. Run the script:

```bash
python contour_detection.py
