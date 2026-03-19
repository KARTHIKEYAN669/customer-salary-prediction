import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data={
    "Experience":[1,2,3,4,5,6,7,8,9,10],
    "Education_Level":[1,2,2,3,3,4,4,5,5,6],
    "Hours_Per_Week":[30,35,40,45,50,40,45,50,55,60],
    "Salary":[20000,25000,30000,40000,50000,55000,60000,70000,80000,90000]
}

df=pd.DataFrame(data)
print(df)

x=df[["Experience","Education_Level","Hours_Per_Week"]]
y=df["Salary"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

error=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error:",error)

error1=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",error1)

new_data=pd.DataFrame([[5,3,45]],columns=["Experience","Education_Level","Hours_Per_Week"])
prediction=model.predict(new_data)
print("Predicted Salary:",prediction)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted")
plt.show()