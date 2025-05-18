from fastapi import UploadFile
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO

async def load_and_preprocess_image(image_file: UploadFile, target_size=(224, 224)):
    image_data = await image_file.read()
    img = load_img(BytesIO(image_data), target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
