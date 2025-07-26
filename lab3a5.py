# importing the necessary packages 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# loading the dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# extracting the feature vectors 
X = df.iloc[:,:196]
Y = df["LABEL"]

#splitting data into training and test set 
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

#training KNN classifier
n = KNeighborsClassifier(n_neighbors=3)
n.fit(X_train,Y_train)