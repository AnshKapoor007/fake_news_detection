import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    body = data.get('body', '')

    # Load the vectorizer and model
    with open('./models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('./models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Preprocess the input data (you need to implement this based on your vectorizer)
    input_vector = vectorizer.transform([body])

    # Make predictions using the loaded model
    prediction = model.predict(input_vector)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5000)