import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

app = FastAPI()

# Load the saved TensorFlow model
model = tf.saved_model.load('savedmodels')

# List of class names (corrected with commas separating each class name)
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

# Confidence threshold (set to 90%)
CONFIDENCE_THRESHOLD = 99.0

# Function to preprocess an image
def load_and_preprocess_image(image_path, target_size=(300, 300)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    # Preprocess the image
    image = load_and_preprocess_image(image_path)
    
    # Make predictions
    infer = model.signatures['serving_default']  # Default signature for inference
    predictions = infer(tf.convert_to_tensor(image))

    # Assuming the output is a dictionary with class probabilities
    probabilities = predictions['output_0'].numpy()[0]  # Extract probabilities
    predicted_class = np.argmax(probabilities)  # Get the predicted class index
    predicted_confidence = probabilities[predicted_class] * 100  # Convert to percentage

    # Remove the temporary file
    os.remove(image_path)
    
    # If confidence is lower than the threshold, reject the prediction
    if predicted_confidence < CONFIDENCE_THRESHOLD:
        response = {
            "message": "Prediction confidence is too low to make a reliable prediction.",
            "hint":"Upload or Click new Image"
        }
    else:
        # Create the response with class names and prediction percentages
        response = {
            "predicted_class": class_names[predicted_class],  # Return the predicted class name
            "predicted_confidence": predicted_confidence,  # Confidence of the predicted class
        }
    
    return JSONResponse(content=response)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
