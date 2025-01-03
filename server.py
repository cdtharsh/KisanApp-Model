import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from io import BytesIO

# Set environment variables for optimized GPU memory usage
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

app = FastAPI()

# Set GPU memory limit if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        print("Memory limit set to 4GB")
    except RuntimeError as e:
        print(e)

# Load the saved TensorFlow model once during startup
model = tf.saved_model.load('savedmodels')

# List of class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Confidence threshold (set to 99%)
CONFIDENCE_THRESHOLD = 90.0

# Function to preprocess an image from UploadFile (in memory)
async def load_and_preprocess_image(image_file: UploadFile, target_size=(224, 224)):
    image_data = await image_file.read()  # Read the image file data
    img = load_img(BytesIO(image_data), target_size=target_size)  # Load image directly from memory
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Preprocess the image
    image = await load_and_preprocess_image(file)

    # Make predictions
    infer = model.signatures['serving_default']  # Default signature for inference
    predictions = infer(tf.convert_to_tensor(image))

    # The output is a dictionary with class probabilities
    probabilities = predictions['output_0'].numpy()[0]  # Extract probabilities
    predicted_class = np.argmax(probabilities)  # Get the predicted class index
    predicted_confidence = probabilities[predicted_class] * 100  # Convert to percentage

    # If confidence is lower than the threshold, reject the prediction
    if predicted_confidence < CONFIDENCE_THRESHOLD:
        response = {
            "message": "Prediction confidence is too low to make a reliable prediction.",
            "hint": "Upload or Click new Image"
        }
    else:
        # Create the response with class names and prediction percentages
        response = {
            "predicted_class": class_names[predicted_class],  # Return the predicted class name
            "predicted_confidence": predicted_confidence,  # Confidence of the predicted class
        }
    
    return JSONResponse(content=response)

# Convert the model to TensorFlow Lite (optional for faster inference)
# def convert_model_to_tflite():
#     converter = tf.lite.TFLiteConverter.from_saved_model('savedmodels')
#     tflite_model = converter.convert()
#     with open('model_quantized.tflite', 'wb') as f:
#         f.write(tflite_model)
#     print("Model successfully converted to TensorFlow Lite format.")

# Run the application
if __name__ == "__main__":
    import uvicorn
    # Optionally convert the model to TFLite format if it's not already done
    # convert_model_to_tflite()  
    uvicorn.run(app, host="localhost", port=8000)
