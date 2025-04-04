import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model_path = "C:/Users/kawal/Desktop/Mini-Project/Plant-Disease-Detectio/models/plant_disease_model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Class labels (ensure they match training dataset)
class_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
                'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
                'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
                'Tomato_healthy']

# ✅ Use absolute path for test image
image_path = "C:/Users/kawal/Desktop/Mini-Project/Plant-Disease-Detectio/test_images/sample3.jpg"

# Check if file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Test image not found at {image_path}")
else:
    print("✅ Test image found!")

# ✅ Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Error: Image not found at {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, (128, 128))

    # Normalize
    image = image / 255.0  

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# ✅ Function to predict disease
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    confidence = np.max(prediction) * 100  # Confidence percentage
    return class_labels[predicted_class], confidence

# Run Prediction
result, confidence = predict_image(image_path)
print(f"🌱 Predicted Disease: {result} ({confidence:.2f}% confidence)")
