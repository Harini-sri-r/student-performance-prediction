from flask import Flask,render_template,request
import joblib
import pandas as pd

app=Flask(__name__)
model=joblib.load('student_marks_predictor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    studytime=float(request.form['studytime'])
    failures=float(request.form['failures'])
    absences=float(request.form['absences'])
    G1=float(request.form['G1'])
    G2=float(request.form['G2'])
    sample_data=pd.DataFrame(
    [[studytime,failures,absences,G1,G2]],
        columns=['studytime','failures','absences','G1','G2']
    )
    prediction=model.predict(sample_data)
    return render_template(
        'index.html',
        prediction_text=f'Predicted Final Marks:{prediction[0]:.2f}')


if __name__=="__main__":
    app.run(debug=True)