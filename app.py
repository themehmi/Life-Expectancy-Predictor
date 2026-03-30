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
        # 2. Extract features from form
        # NOTE: Ensure these keys match the 'name' attribute in your HTML exactly.
        # Order must match the order your model was trained on.
        
        # We use .get(key, default) to prevent KeyError if a slider is missing
        feature_list = [
            float(request.form.get('adult_mortality', 0)),
            float(request.form.get('infant_deaths', 0)),
            float(request.form.get('alcohol', 0)),
            float(request.form.get('percentage_expenditure', 0)),
            float(request.form.get('hepatitis_b', 0)),
            float(request.form.get('measles', 0)),
            float(request.form.get('bmi', 0)),
            float(request.form.get('under_five_deaths', 0)),
            float(request.form.get('polio', 0)),
            float(request.form.get('total_expenditure', 0)),
            float(request.form.get('diphtheria', 0)),
            float(request.form.get('hiv_aids', 0)),
            float(request.form.get('gdp', 0)),
            float(request.form.get('population', 0)),
            float(request.form.get('thinness_1_19', 0)),
            float(request.form.get('thinness_5_9', 0)),
            float(request.form.get('income_composition', 0)),
            float(request.form.get('schooling', 0))
        ]

        # 3. Handle the Categorical 'Status' (Developing vs Developed)
        status_raw = request.form.get('status_developing')
        status_value = 1.0 if status_raw == 'True' else 0.0
        feature_list.append(status_value)

        # 4. Predict
        final_features = np.array(feature_list).reshape(1, -1)
        prediction = model.predict(final_features)
        
        result = round(float(prediction[0]), 2)

        return render_template('predict.html', pred=result)

    except Exception as e:
        # Useful for debugging which key is causing the error
        return render_template('predict.html', pred=f"Input Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
