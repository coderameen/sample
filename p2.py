#one hot encoding
import pandas as pd
import numpy as np

df = pd.read_csv('location.csv')
#print(df)

dummies = pd.get_dummies(df.Nation)
#print(dummies)

#to join dummies columns into original dataset
df_dummies = pd.concat([df,dummies], axis = 'columns')
print(df_dummies)

#to delete columns in df, using drop command

df_dummies.drop(['Nation','america'],axis='columns',inplace=True)
print(df_dummies)