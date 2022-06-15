import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

df = pd.read_csv('insurance.csv')
print(df.head())
'''
plt.scatter(df.age,df.buy_insurance,marker='+',color='red')
plt.show()
'''
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
model = LogisticRegression()
model.fit(x_train,y_train)

prediction = model.predict([[90]])
print(prediction)