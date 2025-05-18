import numpy as np
import tensorflow as tf
from model import infer, class_names
from metadata import class_metadata

def predict_disease(image, confidence_threshold):
    predictions = infer(tf.convert_to_tensor(image))
    probabilities = predictions['output_0'].numpy()[0]

    predicted_class_idx = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_idx].strip()
    predicted_confidence = probabilities[predicted_class_idx] * 100

    if predicted_confidence < confidence_threshold:
        return {
            "message": "Prediction confidence is too low to make a reliable prediction.",
            "hint": "Upload or click a new image with better clarity."
        }

    crop_name = predicted_class_name.split("___")[0].lower()
    crop_health = "healthy" if predicted_class_name.lower().endswith("___healthy") else "unhealthy"
    meta = class_metadata.get(predicted_class_name, {})

    return {
        "crop_health": crop_health,
        "crops": [crop_name],
        "predicted_confidence": predicted_confidence,
        "predicted_diagnoses": [
            {
                "common_name": meta.get("common_name"),
                "scientific_name": meta.get("scientific_name"),
                "diagnosis_likelihood": meta.get("diagnosis_likelihood"),
                "hosts": meta.get("hosts"),
                "pathogen_class": meta.get("pathogen_class"),
                "image_references": meta.get("image_references"),
                "trigger": meta.get("trigger"),
                "identification": {
                    "visual_symptoms": meta.get("identification", {}).get("visual_symptoms", []),
                    "progression": meta.get("identification", {}).get("progression", ""),
                    "differential_diagnosis": meta.get("identification", {}).get("differential_diagnosis", [])
                },
                "preventive_measures": {
                    "cultural_practices": meta.get("preventive_measures", {}).get("cultural_practices", []),
                    "field_monitoring": meta.get("preventive_measures", {}).get("field_monitoring", [])
                },
                "treatment": {
                    "chemical": {
                        "description": meta.get("treatment", {}).get("chemical", {}).get("description"),
                        "chemicals": meta.get("treatment", {}).get("chemical", {}).get("chemicals", []),
                        "notes": meta.get("treatment", {}).get("chemical", {}).get("notes")
                    },
                    "organic": {
                        "description": meta.get("treatment", {}).get("organic", {}).get("description"),
                        "chemicals": meta.get("treatment", {}).get("organic", {}).get("chemicals", []),
                        "notes": meta.get("treatment", {}).get("organic", {}).get("notes")
                    }
                }
            }
        ]
    }
