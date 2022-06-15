import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

df = pd.read_csv('carprediction.csv')
print(df.head())
'''
plt.scatter(df['Mileage'],df['Sell Price'])
plt.show()

'''

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3)

#print(len(x))
#print(len(x_test))

model = LinearRegression()
model.fit(x_train,y_train)

#prediction
prediction = model.predict([[58780,4]])
print("The predcition cost of the car is ",prediction)


acc = model.score(x_test,y_test)
print("The accuracy_score of model is ",acc*100)
print(acc*100)

