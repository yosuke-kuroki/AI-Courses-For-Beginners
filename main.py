# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree

df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Admission%20Chance.csv')
df.head()

y = df['Chance of Admit ']
X = df.drop(['Serial No','Chance of Admit '], axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=2529)

dtr=DecisionTreeRegressor(max_depth=3, random_state=2529)

# train model
dtr.fit(X_train,y_train)

# evaluate the model on training sample
print(dtr.score(X_train,y_train))

print(dtr.score(X_test,y_test))

print(dtr.get_params())

# plot tree
fig,ax = plt.subplots(figsize=(15,10))
final=DecisionTreeRegressor(max_depth=3, random_state=2529)
final.fit(X_train,y_train)
plot_tree(final,feature_names=X.columns,filled=True)