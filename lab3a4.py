# importing the necessary packages 
import pandas as pd
import numpy as np 
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# loading the dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# extracting the feature vectors 
X = df[df["LABEL"] == 2]
Y = df[df["LABEL"] == 99]

#calculating Minkwozki distance 
m_matrix = distance_matrix(X, Y, p=3)
print("Minkwozki distance :",m_matrix)

#plotting the graph
sns.heatmap(m_matrix, cmap='viridis')
plt.show()