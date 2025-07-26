# importing the necessary packages 
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

# loading the dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# taking the require feature 
X = df["LABEL"]
print(np.histogram(X))

# plot of histogram
plt.hist(X,bins=1000,range=(X.min(), X.max()))
plt.show()

#calculating mean and variance 
mean_class =X.mean()
print("Mean : " ,mean_class)

v_class = X.var()
print("Variance :" ,v_class)