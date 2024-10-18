// server.js
const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

let model;
const modelPath = ('C:\\Users\\harsh\\OneDrive\\Desktop\\model\\plant-disease-predictor\\model\\savedmodels');
tf.LayersModel.loadSavedModel(modelPath).then(loadedModel => {
    model = loadedModel;
    console.log('Model loaded successfully');
});

// List of class names for the predictions
const classNames = [
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
];

// Confidence threshold (set to 90%)
const CONFIDENCE_THRESHOLD = 90.0;

// Function to preprocess the uploaded image
const loadAndPreprocessImage = async (imagePath) => {
    const imageBuffer = fs.readFileSync(imagePath);
    const imageTensor = tf.node.decodeImage(imageBuffer)
        .resizeBilinear([300, 300]) // Resize the image to the model's input size
        .div(tf.scalar(255)) // Normalize to [0, 1]
        .expandDims(0); // Add a batch dimension
    return imageTensor;
};

// Prediction endpoint
app.post('/predict', upload.single('file'), async (req, res) => {
    if (!model) {
        return res.status(500).json({ error: 'Model not loaded yet' });
    }

    try {
        const imagePath = req.file.path;

        // Preprocess the image
        const image = await loadAndPreprocessImage(imagePath);
        
        // Make predictions
        const predictions = model.predict(image);
        const probabilities = predictions.dataSync(); // Get class probabilities
        const predictedClassIndex = probabilities.indexOf(Math.max(...probabilities)); // Get the predicted class index
        const predictedConfidence = probabilities[predictedClassIndex] * 100; // Convert to percentage

        // Remove the temporary file
        fs.unlinkSync(imagePath);

        // If confidence is lower than the threshold, reject the prediction
        if (predictedConfidence < CONFIDENCE_THRESHOLD) {
            res.json({
                message: "Prediction confidence is too low to make a reliable prediction.",
                hint: "Upload or click a new image"
            });
        } else {
            // Create the response with class names and prediction percentages
            const response = {
                predicted_class: classNames[predictedClassIndex],
                predicted_confidence: predictedConfidence,
                prediction_percentages: {}
            };

            classNames.forEach((name, index) => {
                response.prediction_percentages[name] = probabilities[index] * 100;
            });

            res.json(response);
        }
    } catch (error) {
        res.status(500).json({ error: 'An error occurred during prediction', details: error.message });
    }
});

// Run the application
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
