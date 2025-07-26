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

# Training KNN classifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train,Y_train)

# Training NN classifier
n = KNeighborsClassifier(n_neighbors=1)
n.fit(X_train,Y_train)

# Accuracy scores of KNN and NN classifier 

print("KNN classifier accuracy :",kn.score(X_test,Y_test))
print("NN classifier accuracy :",n.score(X_test,Y_test))