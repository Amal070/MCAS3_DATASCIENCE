import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

student = pd.read_csv("student_scores.csv")
x = student.iloc[:,0].values.reshape(-1,1)
y = student.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(f"\n MAE : {mae}")
print(f"MSE : {mse}")
print(f"RMSE : {rmse}")

plt.scatter(x,y,color='red',label='data points')
plt.plot(x, model.predict(x), color='blue',label='Regression Line')
plt.title("simple linear regression")
plt.legend()
plt.show()

x_axis = np.arange(len(y_test))
plt.bar(x_axis-0.2,y_test,0.6,label='Actual')
plt.bar(x_axis+0.2,y_pred,0.6,label='Predicted')

plt.xlabel("Test records")
plt.ylabel("Marks")
plt.title("Student score prediction")
plt.legend()
plt.show()
