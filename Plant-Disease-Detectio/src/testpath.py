import cv2
import os

image_path = "C:/Users/kawal/Desktop/Mini-Project/Plant-Disease-Detectio/test_images/sample.jpg"

# Check if the file exists
if os.path.exists(image_path):
    print("✅ Image file exists.")
else:
    print("❌ Image file does NOT exist.")

# Try loading the image
image = cv2.imread(image_path)
if image is None:
    print("❌ OpenCV could NOT read the image.")
else:
    print("✅ OpenCV successfully read the image.")
