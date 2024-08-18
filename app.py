from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.utils import register_keras_serializable
from flask import url_for
import numpy as np

app = Flask(__name__)

# Load the Keras model
@register_keras_serializable
def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true), axis=-1)

# Load the Keras model
model = load_model('model.h5', custom_objects={'mse': mse})


# Define the route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['post'])
def predict():
    # Get data from form submission
    input_data1 = request.form['area']
    input_data2 = request.form['height']
    
    # Perform prediction
    # Assuming model.predict returns an array of predictions
    input_data = np.array([[float(input_data1), float(input_data2), 0, 0, 0, 0, 0, 0, 0]])   
    prediction_array = model.predict(input_data)[0][0]
    
    # Prepare response
    # prediction_values = prediction_array.tolist()  # Convert numpy array to list
    # output = round(prediction[0])
    
    return render_template('index.html', prediction=format(prediction_array))
    

if __name__ == '__main__':
    app.run(debug=True)
