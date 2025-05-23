import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify
import os
import pathlib
import textwrap
import google.generativeai as genai
import requests 
import warnings
warnings.filterwarnings('ignore') 
  
# model = load_model('best_modelv2')
model = load_model('best_modelv4')
def get_precaution(name):    
    genai.configure(api_key='AIzaSyCEYbDu9Bfoq4cKkm_IjRgyWa97h4nnnUU')
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("give me in 10 points What is the precaution for the plant disease " + name)
    data = response.text
    print(type(data))
    # Remove the unwanted '*' and '>' characters before ''
    data = data.replace('> *', '').replace('*', '')

    return data


def preprocess_image(image_path, img_size=(224, 224)):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_images(model, image_paths):
    classes = []
    for path in image_paths:
        preprocessed_img = preprocess_image(path)
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction)
        classes.append(predicted_class)
    return classes

class_names = [
    'Apple__black_rot',
    'Apple__healthy',
    'Apple__rust',
    'Apple__scab',
    'Cassava__bacterial_blight',
    'Cassava__brown_streak_disease',
    'Cassava__green_mottle',
    'Cassava__healthy',
    'Cassava__mosaic_disease',
    'Cherry__healthy',
    'Cherry__powdery_mildew',
    'Chili__healthy',
    'Chili__leaf curl',
    'Chili__leaf spot',
    'Chili__whitefly',
    'Chili__yellowish',
    'Coffee__cercospora_leaf_spot',
    'Coffee__healthy',
    'Coffee__red_spider_mite',
    'Coffee__rust',
    'Corn__common_rust',
    'Corn__gray_leaf_spot',
    'Corn__healthy',
    'Corn__northern_leaf_blight',
    'Cucumber__diseased',
    'Cucumber__healthy',
    'Gauva__diseased',
    'Gauva__healthy',
    'Grape__black_measles',
    'Grape__black_rot',
    'Grape__healthy',
    'Grape__leaf_blight_(isariopsis_leaf_spot)',
    'Jamun__diseased',
    'Jamun__healthy',
    'Lemon__diseased',
    'Lemon__healthy',
    'Mango__diseased',
    'Mango__healthy',
    'Peach__bacterial_spot',
    'Peach__healthy',
    'Pepper_bell__bacterial_spot',
    'Pepper_bell__healthy',
    'Pomegranate__diseased',
    'Pomegranate__healthy',
    'Potato__early_blight',
    'Potato__healthy',
    'Potato__late_blight',
    'Rice__brown_spot',
    'Rice__healthy',
    'Rice__hispa',
    'Rice__leaf_blast',
    'Rice__neck_blast',
    'Soybean__bacterial_blight',
    'Soybean__caterpillar',
    'Soybean__diabrotica_speciosa',
    'Soybean__downy_mildew',
    'Soybean__healthy',
    'Soybean__mosaic_virus',
    'Soybean__powdery_mildew',
    'Soybean__rust',
    'Soybean__southern_blight',
    'Strawberry___leaf_scorch',
    'Strawberry__healthy',
    'Sugarcane__bacterial_blight',
    'Sugarcane__healthy',
    'Sugarcane__red_rot',
    'Sugarcane__red_stripe',
    'Sugarcane__rust',
    'Tea__algal_leaf',
    'Tea__anthracnose',
    'Tea__bird_eye_spot',
    'Tea__brown_blight',
    'Tea__healthy',
    'Tea__red_leaf_spot',
    'Tomato__bacterial_spot',
    'Tomato__early_blight',
    'Tomato__healthy',
    'Tomato__late_blight',
    'Tomato__leaf_mold',
    'Tomato__mosaic_virus',
    'Tomato__septoria_leaf_spot',
    'Tomato__spider_mites_(two_spotted_spider_mite)',
    'Tomato__target_spot',
    'Tomato__yellow_leaf_curl_virus',
    'Wheat__brown_rust',
    'Wheat__healthy',
    'Wheat__septoria',
    'Wheat__yellow_rust'
]

def make_prediction(image_path):
    image_paths = [image_path]
    predicted_classes = classify_images(model,image_paths)
    return class_names[predicted_classes[0]]

print("Trying to start flask app")
# Flask app Start's here 
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    #Create the 'temp' directory if it doesn't exist
    temp_dir = os.path.join(app.root_path, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Save the image file to the 'temp' directory
    temp_image_path = os.path.join(temp_dir, image_file.filename)
    image_file.save(temp_image_path)

    prediction = make_prediction(temp_image_path)

    os.remove(temp_image_path)

    print("The prediction is: ",prediction)

    if "healthy" in prediction:
        return jsonify({
            'prediction': prediction
            })
    else:
         #get precaution's
        payload = {'disease': prediction}
        response = requests.post("http://localhost:5001", json=payload)
        result = response.text
        print(result)
        precaution = result
        # Return the prediction as a response
        return jsonify({
            'prediction': prediction,
            'precautions': precaution
            })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4040)