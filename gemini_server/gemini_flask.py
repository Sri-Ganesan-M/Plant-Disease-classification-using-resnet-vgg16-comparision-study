from flask import Flask, request, jsonify
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore') 
  
app = Flask(__name__)

def get_precaution(name):
    genai.configure(api_key='')
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("give me in 10 points What is the precaution for the plant disease " + name)
    data = response.text
    print(type(data))
    # Remove the unwanted '*' and '>' characters before ''
    data = data.replace('> *', '').replace('*', '')

    return data

@app.route('/', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    # Access the disease name from the JSON data
    disease_name = data.get('disease')
    precaution = get_precaution(disease_name)
    print(precaution)
    return jsonify({'precautions': precaution})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
