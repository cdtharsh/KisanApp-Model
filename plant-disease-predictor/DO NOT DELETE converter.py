import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.applications import Xception
import os  # Import os to check for existing files

# Define your model architecture
# def create_model():
#     # Load the base Xception model
#     base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
#     # Build the model
#     model = Sequential([
#         base_model,
#         Dense(256, activation='relu'),
#         Dropout(0.5),
#         Dense(38, activation='softmax')  # Adjust number of classes as needed
#     ])
    
#     return model
def create_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(38, activation='softmax')  # Adjust number of classes as needed
    ])
    
    return model


# Create a new model
model = create_model()

# Build the model to ensure it's constructed properly
model.build((None, 224, 224, 3))  # Specify the input shape

# Load weights from the saved weights file
weights_path = 'C:\\Users\\harsh\\OneDrive\\Desktop\\model.2.10\\legacy\\model.weights.h5'  # Update this to your actual weights file path

# Attempt to load the weights
try:
    model.load_weights(weights_path)
    print("Weights loaded successfully!")
except Exception as e:
    print("Error loading weights:", e)

# Compile the model before saving
learning_rate_finetune = 0.00001
model.compile(optimizer=Adam(learning_rate=learning_rate_finetune), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the file path for saving the model
model_save_path = 'converted_my_model.keras'  # Use the native Keras format

# Check if the file already exists
if os.path.exists(model_save_path):
    print(f"File {model_save_path} already exists. Deleting the old file.")
    os.remove(model_save_path)  # Delete the old model file if it exists

# Save the model in the Keras format
model.save(model_save_path)  # No need for save_format argument in Keras 3

print("Model saved successfully in Keras format!")
