from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('heart1.csv')
print(df.head())

x = df.iloc[:,:-1]#iloc[rows,column]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=10)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

prediction = model.predict([[71,0,0,112,149,0,1,125,0,1.6,1,0,2]])
#print("The prediction is ",prediction)

if prediction == 1:
	print("The patient has Heart Disease")
else:
	print("The patient doesn't have heart Disease")

#model accuracy 

y_pred = model.predict(x_test)
acc = accuracy_score(y_pred,y_test)
print("the accuracy of the model is ",acc*100)

#writing the trained model inside the variable
file = open('model_pkl','wb')#
pickle.dump(model,file)

file = open('model_pkl','rb')
mp = pickle.load(file)

print(mp.predict([[50,0,1,120,244,0,1,162,0,1.1,2,0,2]]))





