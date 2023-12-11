from flask import Flask, render_template, request
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np


# How the algorithm works:
# *The structure of the Bayesian Network is defined using the BayesianNetwork class from the pgmpy.models module.
# *The network structure specifies how different symptoms are probabilistically related to each other.
# **The synthetic dataset is generated, and then the Bayesian Network is trained (fitted) using the Maximum Likelihood Estimation method. 
# **This is done with the fit method of the Bayesian Network model, and the MaximumLikelihoodEstimator is used as the estimator.
# ***Variable Elimination (VariableElimination class from pgmpy.inference) is set up for performing inference in the Bayesian Network.
# ***This will be used later to query the network and make predictions based on observed evidence.

# More Information of Used Algorithm:
# The key components involved in Bayesian Network implementation include defining the network structure, generating a dataset for training, 
# and then fitting the model to the data.
#  Afterward, the trained model is used for inference, and in this specific project, it's employed for diagnosing illnesses
#  based on observed symptoms.

# Library REFERENCE: https://pgmpy.org/param_estimator/mle.html
# REFERENCE: https://flask.palletsprojects.com/en/3.0.x/
# https://github.com/JaavLex/shootingclubpy
# https://realpython.com/tutorials/flask/

# REFERENCE: https://numpy.org/doc/stable/

app = Flask(__name__)

# IMPORTANT RESEARCH PAPER: https://www.cis.upenn.edu/~mkearns/papers/barbados/heckerman.pdf
# Note: We have learned how to create synthetic_dataset by looking through Page_36 code example structure

# ***EDGES
model = BayesianNetwork([('Flu', 'Fever'), ('Flu', 'Fatigue'), ('Fever', 'Cough'), ('Fatigue', 'Cough'),
                         ('Flu', 'Headache'), ('Flu', 'SoreThroat'), ('Fever', 'BodyAche'), ('Fatigue', 'BodyAche'),
                         ('Cough', 'ChestPain'), ('BodyAche', 'ChestPain'), ('Flu', 'Chills'), ('Flu', 'Nausea'),
                         ('Flu', 'ShortnessOfBreath'), ('Flu', 'LossOfTaste'), ('Flu', 'LossOfSmell')])

np.random.seed(42)
num_samples = 1000

# General Information:
'''
A Bayesian Network is a probabilistic graphical model that represents probabilistic relationships among a set of variables.
In this code, the Bayesian Network is represented using the pgmpy library, 
where nodes correspond to symptoms, and edges represent probabilistic dependencies between them.
'''

'''
In Case of Question:
What is Maximum Likelihood Estimation, and why is it used in training the Bayesian Network?

 Maximum Likelihood Estimation (MLE) is a method used to estimate the parameters of a statistical model. 
 In this context, MLE is used to estimate the conditional probability distributions of symptoms given the presence or absence of the flu. 
 It is employed to maximize the likelihood of observing the given dataset under the assumed model.
'''


# ***NODES
columns = ['Flu', 'Fever', 'Fatigue', 'Cough', 'Headache', 'SoreThroat', 'BodyAche', 'ChestPain', 'Chills', 'Nausea',
           'ShortnessOfBreath', 'LossOfTaste', 'LossOfSmell']

# ***We have created an empty DataFrame to store the synthetic dataset
data = pd.DataFrame(columns=columns)

# https://stackoverflow.com/questions/43664994/is-there-a-difference-between-the-input-paramaters-of-numpy-random-choice-and-ra
# https://stackoverflow.com/questions/73724150/problem-with-np-random-multinomial-and-size-option-in-python?rq=2
# https://www.w3schools.com/jsref/jsref_random.asp
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
# https://arxiv.org/abs/cs/9501101
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

# ***We set up Variable Elimination for inference in the Bayesian Network
inference = VariableElimination(model)

# REFERENCE: https://flask.palletsprojects.com/en/2.1.x/quickstart/#rendering-templates
@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/')
def index():
    return render_template('index.html', symptoms=columns[1:])


@app.route('/diagnose', methods=['POST'])
def diagnose():
    symptoms = request.form.getlist('symptoms')
    if not symptoms:
        return render_template('diagnosis.html', diagnosis="No illnesses selected", probability=None)

# ***Perform inference to diagnose the likelihood of having the flu
    query_result = inference.query(variables=['Flu'], evidence=dict(zip(symptoms, [1]*len(symptoms))))
    probability = query_result.values[1]

    diagnosis = "Covid-19" if probability > 0.5 else "Influenza"
# https://www.digitalocean.com/community/tutorials/how-to-use-templates-in-a-flask-application

# ***Render the diagnosis page with the result
    return render_template('diagnosis.html', diagnosis=diagnosis, probability=probability)



if __name__ == '__main__':
    data = pd.read_csv('synthetic_dataset_complex.csv')

    model.fit(data, estimator=MaximumLikelihoodEstimator)

    inference = VariableElimination(model)

    app.run(debug=True)