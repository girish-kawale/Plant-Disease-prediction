import kagglehub
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get dataset path
dataset_base_path = kagglehub.dataset_download("emmarex/plantdisease")
dataset_path = os.path.join(dataset_base_path, "PlantVillage")  # Go inside "PlantVillage" folder

# Remove the extra "PlantVillage" folder inside if it exists
extra_folder_path = os.path.join(dataset_path, "PlantVillage")
if os.path.exists(extra_folder_path):
    shutil.rmtree(extra_folder_path)  # Delete the extra folder
    print("Removed extra 'PlantVillage' folder inside dataset.")

print("Final Dataset Path:", dataset_path)
print("Classes available:", os.listdir(dataset_path))  # Should list only disease categories

# Image settings
IMG_SIZE = (128, 128)  
BATCH_SIZE = 32

# Data Augmentation and Normalization
datagen = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2  
)

# Load Training Data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# Load Validation Data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("Fixed Classes in dataset:", train_data.class_indices)  # PlantVillage should not appear
