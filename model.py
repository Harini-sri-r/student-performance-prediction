import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
data=pd.read_csv('dataset/student_data.csv')
X=data[['studytime','failures','absences','G1','G2']]
y=data['G3']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
error=mean_absolute_error(y_test,predictions)
print("Mean Absolute Error:",error)
studytime=float(input("Enter studytime: "))
failures=float(input("Enter failures:"))
absences=float(input("Enter absences:"))
G1=float(input("Enter G1 marks:"))
G2=float(input("Enter G2 marks:"))
sample_data=pd.DataFrame(
    [[studytime,failures,absences,G1,G2]],
    columns=['studytime','failures','absences','G1','G2']
)

prediction_marks=model.predict(sample_data)
print("Predicted Final Marks",prediction_marks[0])
accuracy=r2_score(y_test,predictions)
print("R2-score",accuracy)
plt.scatter(y_test,predictions)
plt.plot([0,20],[0,20])
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()

import joblib
joblib.dump(model,'student_marks_predictor.pkl')