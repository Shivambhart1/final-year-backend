import cv2
import numpy as np

# Corrected disease list with all classes properly separated by commas
diseaseList = [
    "Actinic Keratosis Intraepithelical Carcinoma",
    "Basal Cell Carcinoma",
    "Dermatofibroma",
    "Melanocytic Nevi",
    "Melanoma",
    "Vascular Lesion",
    "Pseudophakic Bullous Keratopathy (PBK)",
    "Systemic Sclerosis (SSC)",
    "Seborrheic Keratosis (SK)"
]

def getDiseaseNameFromPrediction(prediction_list):
    classIndex = np.argmax(prediction_list)  
    return diseaseList[classIndex]  


def prepareImage(image_path):
    IMG_SIZE = 64  
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  
