from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 1. Load the model once at startup
try:
    model = joblib.load('model.joblib')
except Exception as e:
    print(f"Error: Model file 'model.joblib' not found. {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Processes form data and renders the result page."""
    try:
        # 2. Extract numeric features from form
        # We use a list to maintain the exact order your model expects
        feature_list = [
            float(request.form['adult_mortality']),
            float(request.form['infant_deaths']),
            float(request.form['alcohol']),
            float(request.form['percentage_expenditure']),
            float(request.form['hepatitis_b']),
            float(request.form['measles']),
            float(request.form['bmi']),
            float(request.form['under_five_deaths']),
            float(request.form['polio']),
            float(request.form['total_expenditure']),
            float(request.form['diphtheria']),
            float(request.form['hiv_aids']),
            float(request.form['gdp']),
            float(request.form['population']),
            float(request.form['thinness_1_19']),
            float(request.form['thinness_5_9']),
            float(request.form['income_composition']),
            float(request.form['schooling'])
        ]

        # 3. Handle the Categorical 'Status' (Developing vs Developed)
        # Convert the string "True"/"False" from HTML to 1.0/0.0
        status_raw = request.form.get('status_developing')
        status_value = 1.0 if status_raw == 'True' else 0.0
        feature_list.append(status_value)


        final_features = np.array(feature_list).reshape(1, -1)
        prediction = model.predict(final_features)
        
        result = round(float(prediction[0]), 2)

        return render_template('predict.html', pred=result)

    except Exception as e:
        return render_template('predict.html', pred=f"Input Error: {str(e)}")

if __name__ == '__main__':
    # Run on port 5000 in debug mode for development
    app.run(debug=True, host="0.0.0.0", port=5000)