from flask import Flask, render_template, request
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np

app = Flask(__name__)

# Define a more complex structure for the Bayesian Network
model = BayesianNetwork([('Flu', 'Fever'), ('Flu', 'Fatigue'), ('Fever', 'Cough'), ('Fatigue', 'Cough'),
                       ('Flu', 'Headache'), ('Flu', 'SoreThroat'), ('Fever', 'BodyAche'), ('Fatigue', 'BodyAche'),
                       ('Cough', 'ChestPain'), ('BodyAche', 'ChestPain')])

# Generate a larger synthetic dataset for training
np.random.seed(42)
num_samples = 1000
columns = ['Flu', 'Fever', 'Fatigue', 'Cough', 'Headache', 'SoreThroat', 'BodyAche', 'ChestPain']
data = pd.DataFrame(columns=columns)

for _ in range(num_samples):
    flu = np.random.choice([0, 1])
    fever_prob = 0.8 if flu == 1 else 0.2
    fatigue_prob = 0.7 if flu == 1 else 0.3
    cough_prob = 0.6 if flu == 1 else 0.4
    headache_prob = 0.5 if flu == 1 else 0.5
    sore_throat_prob = 0.4 if flu == 1 else 0.6
    body_ache_prob = 0.6 if flu == 1 else 0.4
    chest_pain_prob = 0.3 if flu == 1 else 0.7

    fever = np.random.choice([0, 1], p=[1 - fever_prob, fever_prob])
    fatigue = np.random.choice([0, 1], p=[1 - fatigue_prob, fatigue_prob])
    cough = np.random.choice([0, 1], p=[1 - cough_prob, cough_prob])
    headache = np.random.choice([0, 1], p=[1 - headache_prob, headache_prob])
    sore_throat = np.random.choice([0, 1], p=[1 - sore_throat_prob, sore_throat_prob])
    body_ache = np.random.choice([0, 1], p=[1 - body_ache_prob, body_ache_prob])
    chest_pain = np.random.choice([0, 1], p=[1 - chest_pain_prob, chest_pain_prob])

    data.loc[data.shape[0]] = {'Flu': flu, 'Fever': fever, 'Fatigue': fatigue, 'Cough': cough,
                           'Headache': headache, 'SoreThroat': sore_throat, 'BodyAche': body_ache, 'ChestPain': chest_pain}


# Save the synthetic dataset to a CSV file
data.to_csv('synthetic_dataset_complex.csv', index=False)

# Estimate CPDs (Conditional Probability Distributions)
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Doing exact inference using Variable Elimination
inference = VariableElimination(model)


@app.route('/')
def index():
    return render_template('index.html', symptoms=columns[1:])


@app.route('/diagnose', methods=['POST'])
def diagnose():
    symptoms = request.form.getlist('symptoms')

    # Query for the probability of having the flu given symptoms
    query_result = inference.query(variables=['Flu'], evidence=dict(zip(symptoms, [1]*len(symptoms))))
    probability = query_result.values[1]

    diagnosis = "Likely Flu" if probability > 0.5 else "Unlikely Flu"

    return render_template('diagnosis.html', diagnosis=diagnosis, probability=probability)


if __name__ == '__main__':
    # Load the synthetic dataset from the CSV file
    data = pd.read_csv('synthetic_dataset_complex.csv')

    # Estimate CPDs (Conditional Probability Distributions)
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Doing exact inference using Variable Elimination
    inference = VariableElimination(model)

    app.run(debug=True)
