from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('models/fraud_detection_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the form data
    time = float(request.form['Time'])
    amount = float(request.form['Amount'])
    
    # Make predictions using the trained model
    prediction = model.predict([[time, amount]])
    
    # Prepare the prediction result message
    if prediction[0] == 1:
        result_message = "Fraudulent transaction detected."
    else:
        result_message = "Transaction is normal."
    
    # Render the result template with the prediction result
    return render_template('result.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)

