from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an HTML file for the form

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    no_of_dependents = request.form['no_of_dependents']
    education = request.form['education']
    self_employed = request.form['self_employed']
    income_annum = request.form['income_annum']
    loan_amount = request.form['loan_amount']
    loan_term = request.form['loan_term']
    cibil_score = request.form['cibil_score']
    residential_assets_value = request.form['residential_assets_value']
    commercial_assets_value = request.form['commercial_assets_value']
    luxury_assets_value = request.form['luxury_assets_value']
    bank_asset_value = request.form['bank_asset_value']

    # Mapping categorical data to numerical values
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}

    # Prepare the input data for prediction
    input_data = np.array([[int(no_of_dependents),
                            education_map[education],
                            self_employed_map[self_employed],
                            float(income_annum),
                            float(loan_amount),
                            float(loan_term),
                            float(cibil_score),
                            float(residential_assets_value),
                            float(commercial_assets_value),
                            float(luxury_assets_value),
                            float(bank_asset_value)]])

    # Predict using the loaded model
    prediction = model.predict(input_data)

    # Result
    result = "Approved" if prediction[0] == 1 else "Rejected"

    # Render the result page
    return render_template('result.html', 
                           prediction_text=f'Loan Status: {result}',
                           no_of_dependents=no_of_dependents,
                           education=education,
                           self_employed=self_employed,
                           income_annum=income_annum,
                           loan_amount=loan_amount,
                           loan_term=loan_term,
                           cibil_score=cibil_score,
                           residential_assets_value=residential_assets_value,
                           commercial_assets_value=commercial_assets_value,
                           luxury_assets_value=luxury_assets_value,
                           bank_asset_value=bank_asset_value)

if __name__ == '__main__':
    app.run(debug=True)
