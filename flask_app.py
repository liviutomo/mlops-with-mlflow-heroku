import pickle
from flask import Flask, request, jsonify

# Load the model
model = pickle.load(open('model/model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Machine Learning Model API!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=False)
