from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    result = "Good Wine 🍷✅" if prediction == 1 else "Not Good Wine ❌"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
