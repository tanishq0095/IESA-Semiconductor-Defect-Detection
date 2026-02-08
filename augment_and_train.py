import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tf2onnx

# 1. SETUP PATHS
# Ensure this folder contains EXACTLY 8 subfolders: 
# clean, bridge, scratch, crack, short, open, malformed_via, other
dataset_path = r'C:\Users\mange\OneDrive\Desktop\ic_hackathon\dataset'

# 2. DATA AUGMENTATION (Ensures 1,000+ images for better performance)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2  # 20% for testing
)

# Load training data
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

# Load validation data
val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

# 3. BUILD LIGHTWEIGHT MODEL (Optimized for NXP i.MX RT Edge Devices)
# We use a custom CNN head on MobileNetV2 for efficiency
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 1), 
    include_top=False, 
    weights=None
)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(8, activation='softmax') # EXACTLY 8 CLASSES
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. TRAINING
print("\nStarting Training... Building your Edge-AI solution.")
model.fit(train_gen, validation_data=val_gen, epochs=10)

# 5. SAVE KERAS MODEL FIRST (Safety Step)
model.save("temp_model.keras")

# 6. CONVERT TO ONNX
print("\nConverting model to ONNX format...")
# This method is more stable for Sequential models
os.system("python -m tf2onnx.convert --keras temp_model.keras --output semiconductor_defect_model.onnx --opset 13")

print("\nCHECK YOUR FOLDER: If 'semiconductor_defect_model.onnx' is there, you are ready!")