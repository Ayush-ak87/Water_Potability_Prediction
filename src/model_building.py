import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("./data/processed/train_processed.csv")

# x_train = train_data.iloc[:,0:-1].values
# y_train = train_data.iloc[:,-1].values

x_train = train_data.drop(columns=["Potability"], axis=1)
y_train = train_data["Potability"]

model = RandomForestClassifier()
model.fit(x_train,y_train)

pickle.dump(model,open("model.pkl","wb"))