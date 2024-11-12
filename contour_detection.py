import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images (adjust file paths based on your folder structure)
image1 = cv2.imread('images/image1.jpg')
image2 = cv2.imread('images/image2.jpg')
image3 = cv2.imread('images/image3.jpg')
image4 = cv2.imread('images/image4.jpg')

# Function to apply edge detection and contour extraction
def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Canny edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    
    return contour_image

# Detect contours on all images
contour_image1 = detect_contours(image1)
contour_image2 = detect_contours(image2)
contour_image3 = detect_contours(image3)
contour_image4 = detect_contours(image4)

# Display the results
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(contour_image1, cv2.COLOR_BGR2RGB))
plt.title("Contours - Simple Shapes")
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(contour_image2, cv2.COLOR_BGR2RGB))
plt.title("Contours - Complex Objects")
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(contour_image3, cv2.COLOR_BGR2RGB))
plt.title("Contours - Noisy Image")
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(contour_image4, cv2.COLOR_BGR2RGB))
plt.title("Contours - Real-World Scene")

plt.tight_layout()
plt.show()
