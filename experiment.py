import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from PIL import Image

# Function to create the model
def create_model():
    # Load the base Xception model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    # Build the model
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

# Compile the model
learning_rate_finetune = 0.00001
model.compile(optimizer=Adam(learning_rate=learning_rate_finetune), loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights from the saved weights file
weights_path = 'C:\\Users\\harsh\\OneDrive\\Desktop\\model.2.10\\legacy\\model.weights.h5'  # Update this to your actual weights file path
model.load_weights(weights_path)

# Save the model in Keras format
model.save('converted_my_model.keras')

# Function to prepare an image for prediction
def prepare_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Example usage for making predictions
image_path = ''  # Replace with your image path
input_data = prepare_image(image_path)

# Make predictions
predictions = model.predict(input_data)
predicted_class = np.argmax(predictions, axis=1)

print(f"Predicted class: {predicted_class[0]}")
