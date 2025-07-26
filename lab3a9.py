# importing the necessary packages 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , precision_score , recall_score , f1_score , accuracy_score

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
Y_train_pred = n.predict(X_train)
Y_test_pred = n.predict(X_test)

#confusion matrix 
cm = confusion_matrix(Y_test,Y_test_pred)
print(cm)

#Performance metrics
train_precision = precision_score(Y_train,Y_train_pred, average="weighted")
train_recall = recall_score(Y_train,Y_train_pred, average="weighted")
train_f1 = f1_score(Y_train,Y_train_pred, average="weighted")

test_precision = precision_score(Y_test,Y_test_pred, average="weighted")
test_recall = recall_score(Y_test,Y_test_pred, average="weighted")
test_f1 = f1_score(Y_test,Y_test_pred, average="weighted")

print("Training precision :",train_precision)
print("Training recall :",train_recall)
print("Training F1-score :",train_f1)

print("Test precision :",test_precision)
print("Test recall :",test_recall)
print("Test F1-score :",test_f1)

#finding the fit of the model

train_accuracy = accuracy_score(Y_train,Y_train_pred)
test_accuracy = accuracy_score(Y_test,Y_test_pred)

print("Training Accuracy :",train_accuracy)
print("Testing Accuracy :",test_accuracy)