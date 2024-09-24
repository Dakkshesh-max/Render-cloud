from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained LightGBM model
model_path = 'lightgbm_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form submission
    order_number = float(request.form['order_number'])
    days_since_prior_order = float(request.form['days_since_prior_order'])
    add_to_cart_order = float(request.form['add_to_cart_order'])
    recency = float(request.form['recency'])
    frequency = float(request.form['frequency'])
    monetary = float(request.form['monetary'])
    reorder_rate = float(request.form['reorder_rate'])

    # Prepare input data for prediction
    input_data = np.array([[order_number, days_since_prior_order, add_to_cart_order, recency, frequency, monetary, reorder_rate]])

    # Make prediction using the locally loaded model
    prediction = model.predict(input_data, predict_disable_shape_check=True)


    # Interpret the result
    prediction_text = "Reorder Prediction: Yes" if prediction[0] > 0.5 else "Reorder Prediction: No"
    
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
