# importing the necessary packages 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

# loading the dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# extracting the feature vectors 
X = df.iloc[:,:196]
Y = df["LABEL"]

print(X)
print(Y)

#splitting data into training and test set 
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

#printing training and test set 

print("Training set:")
print(X_train)
print(Y_train)

print("Test set:")
print(X_test)
print(Y_test)