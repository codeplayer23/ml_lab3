# importing the necessary packages 
import pandas as pd 
import numpy as np 

# loading the dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# printing the mean of each class 
mean_class = df.groupby("LABEL").mean()
print(mean_class)

#printing the standard deviation of each class
sd_class = df.groupby("LABEL").std()
print(sd_class)

#distance between classes 
dist = mean_class.apply(lambda row : np.linalg.norm(row) , axis = 1)
print(dist)