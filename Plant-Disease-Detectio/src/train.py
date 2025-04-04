import os
import tensorflow as tf
from data_loader import train_data, val_data
from model import build_model

# Get number of classes
num_classes = len(train_data.class_indices)

# Build model
model = build_model(num_classes)

# Train the model
EPOCHS = 10  # You can increase this for better accuracy
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save the trained model
model_save_path = os.path.join(os.getcwd(), "models", "plant_disease_model.h5")


os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")
