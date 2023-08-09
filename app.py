from flask import Flask, render_template, request
import os 
import numpy as np

from South_German_Bank.pipeline.prediction import PredictionPipeline


app = Flask(__name__)

@app.route('/', methods=['GET'])

def homepage():
    return render_template('index.html')


@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successfully Completed !!"


@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method  == "POST":
        try:
            # Read the input given by the user

            duration = int(request.form('duration'))
            amount = int(request.form('amount'))
            age = int(request.form('age'))
            status = str(request.form('status'))
            credit_history = str(request.form('credit_history'))
            purpose = str(request.form('purpose'))
            savings = str(request.form('savings'))
            employment_duration = str(request.form('employment_duration'))
            installment_rate = str(request.form('installment_rate'))
            personal_status_sex = str(request.form('personal_status_sex'))
            other_debtors = str(request.form('other_debtors'))
            present_residence = str(request.form('present_residence'))
            property = str(request.form('property'))
            other_installment_plans = str(request.form('other_installment_plans'))
            housing = str(request.form('housing'))
            number_credits = str(request.form('number_credits'))
            job = str(request.form('job'))
            people_liable = str(request.form('people_liable'))
            telephone = str(request.form('telephone'))
            foreign_worker = str(request.form('foreign_worker'))

            data = [duration, amount, age, status, credit_history, purpose,
                    savings, employment_duration, installment_rate, personal_status_sex,
                    other_debtors, present_residence, property, other_installment_plans, housing, 
                    number_credits, job, people_liable, telephone, foreign_worker]
            
            data = np.array(data).reshape(1, 20)

            obj = PredictionPipeline()
            predict = obj.predict(data)
            print("Prediction:", predict)  # Add this line to check the value of predict
            return render_template('results.html', prediction=str(predict))
        

        except Exception as e:
            print('The Exception message is: ', e)
            return "Something Went Wrong"
        

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
