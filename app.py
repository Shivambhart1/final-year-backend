from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from models.runModel import prepareImage, getDiseaseNameFromPrediction
import os

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# Ensure the uploads folder exists
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Loading the model
model = tf.keras.models.load_model('models/Skin_disease.h5') 

@app.route("/", methods=['GET'])
def home():
    return "Server is running"

@app.route("/api/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    imageFile = request.files['file']
    if imageFile.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded file
        imagePath = os.path.join(UPLOAD_FOLDER, imageFile.filename)
        imageFile.save(imagePath)

        # Prepare the image for the model
        preparedImage = prepareImage(imagePath)


        predictions = model.predict(preparedImage).tolist()[0]
        diseaseName = getDiseaseNameFromPrediction(predictions)

        response = {
            "disease_name": diseaseName,
            "predictions": predictions  
        }
        print(response)
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Error processing file/(s): {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
