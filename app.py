# importing libraries
import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        try:
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list)
            if int(result) == 0:
                prediction: str = 'Iris setosa'
            elif int(result) == 1:
                prediction = 'Iris virginica'
            elif int(result) == 2:
                prediction = 'Iris versicolor'
            else:
                prediction = f'{int(result)} Not defined'
        except ValueError:
            prediction = 'Value Error try again'

    return render_template('index.html', prediction_text='The flower is {}'.format(prediction))


if __name__ == "__main__":
  app.run(debug=True)
