# Thomas Delvaux
# ME-6930 036
# 03/24/2021

# https://likegeeks.com/python-correlation-matrix/

# import numpy as np

# np.random.seed(10)

# # generating 10 random values for each of the two variables
# X = np.random.randn(10)

# Y = np.random.randn(10)

# # computing the corrlation matrix
# C = np.corrcoef(X,Y)

# print(C)


###
from sklearn.datasets import load_breast_cancer

import pandas as pd

breast_cancer = load_breast_cancer()

data = breast_cancer.data

features = breast_cancer.feature_names

df = pd.DataFrame(data, columns = features)

print(df.shape)

print(features)


###
import seaborn as sns

import matplotlib.pyplot as plt

# taking all rows but only 6 columns
df_small = df.iloc[:,:6]

correlation_mat = df_small.corr()

sns.heatmap(correlation_mat, annot = True)

plt.title("Correlation matrix of Breast Cancer data")

plt.xlabel("cell nucleus features")

plt.ylabel("cell nucleus features")

plt.show()