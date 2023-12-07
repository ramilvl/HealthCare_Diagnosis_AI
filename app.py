from flask import Flask, render_template, request
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np

app = Flask(__name__)

model = BayesianNetwork([('Flu', 'Fever'), ('Flu', 'Fatigue'), ('Fever', 'Cough'), ('Fatigue', 'Cough'),
                         ('Flu', 'Headache'), ('Flu', 'SoreThroat'), ('Fever', 'BodyAche'), ('Fatigue', 'BodyAche'),
                         ('Cough', 'ChestPain'), ('BodyAche', 'ChestPain'), ('Flu', 'Chills'), ('Flu', 'Nausea'),
                         ('Flu', 'ShortnessOfBreath'), ('Flu', 'LossOfTaste'), ('Flu', 'LossOfSmell')])

np.random.seed(42)
num_samples = 1000
columns = ['Flu', 'Fever', 'Fatigue', 'Cough', 'Headache', 'SoreThroat', 'BodyAche', 'ChestPain', 'Chills', 'Nausea',
           'ShortnessOfBreath', 'LossOfTaste', 'LossOfSmell']
data = pd.DataFrame(columns=columns)

for _ in range(num_samples):
    flu = np.random.choice([0, 1])
    fever_prob = 0.5 if flu == 1 else 0.2
    fatigue_prob = 0.2 if flu == 1 else 0.3
    cough_prob = 0.6 if flu == 1 else 0.4
    headache_prob = 0.5 if flu == 1 else 0.5
    sore_throat_prob = 0.4 if flu == 1 else 0.6
    body_ache_prob = 0.6 if flu == 1 else 0.4
    chest_pain_prob = 0.5 if flu == 1 else 0.5
    chills_prob = 0.8 if flu == 1 else 0.2
    nausea_prob = 0.7 if flu == 1 else 0.3
    shortness_of_breath_prob = 0.6 if flu == 1 else 0.4
    loss_of_taste_prob = 0.5 if flu == 1 else 0.5
    loss_of_smell_prob = 0.4 if flu == 1 else 0.6

    fever = np.random.choice([0, 1], p=[1 - fever_prob, fever_prob])
    fatigue = np.random.choice([0, 1], p=[1 - fatigue_prob, fatigue_prob])
    cough = np.random.choice([0, 1], p=[1 - cough_prob, cough_prob])
    headache = np.random.choice([0, 1], p=[1 - headache_prob, headache_prob])
    sore_throat = np.random.choice([0, 1], p=[1 - sore_throat_prob, sore_throat_prob])
    body_ache = np.random.choice([0, 1], p=[1 - body_ache_prob, body_ache_prob])
    chest_pain = np.random.choice([0, 1], p=[1 - chest_pain_prob, chest_pain_prob])
    chills = np.random.choice([0, 1], p=[1 - chills_prob, chills_prob])
    nausea = np.random.choice([0, 1], p=[1 - nausea_prob, nausea_prob])
    shortness_of_breath = np.random.choice([0, 1], p=[1 - shortness_of_breath_prob, shortness_of_breath_prob])
    loss_of_taste = np.random.choice([0, 1], p=[1 - loss_of_taste_prob, loss_of_taste_prob])
    loss_of_smell = np.random.choice([0, 1], p=[1 - loss_of_smell_prob, loss_of_smell_prob])

    data.loc[data.shape[0]] = {'Flu': flu, 'Fever': fever, 'Fatigue': fatigue, 'Cough': cough,
                               'Headache': headache, 'SoreThroat': sore_throat, 'BodyAche': body_ache,
                               'ChestPain': chest_pain, 'Chills': chills, 'Nausea': nausea,
                               'ShortnessOfBreath': shortness_of_breath, 'LossOfTaste': loss_of_taste,
                               'LossOfSmell': loss_of_smell}

data.to_csv('synthetic_dataset_complex.csv', index=False)

model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)


@app.route('/')
def index():
    return render_template('index.html', symptoms=columns[1:])


@app.route('/diagnose', methods=['POST'])
def diagnose():
    symptoms = request.form.getlist('symptoms')

    query_result = inference.query(variables=['Flu'], evidence=dict(zip(symptoms, [1]*len(symptoms))))
    probability = query_result.values[1]

    diagnosis = "Covid-19" if probability > 0.5 else "Influenza"

    return render_template('diagnosis.html', diagnosis=diagnosis, probability=probability)


if __name__ == '__main__':
    data = pd.read_csv('synthetic_dataset_complex.csv')

    model.fit(data, estimator=MaximumLikelihoodEstimator)

    inference = VariableElimination(model)

    app.run(debug=True)
