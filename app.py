from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('D:\Credit Card Fraud\fraud_detection_model.pkl' , 'rb'))

model 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


    input_data = []
    for feature in features:
        input_data.append(float(request.form.get(feature)))

    input_df = pd.DataFrame([input_data], columns=features)

    prediction = model.predict_proba(input_df)[:, 1]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
