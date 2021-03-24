# Thomas Delvaux
# ME-6930 036
# 03/24/2021

# Correlation code based on:
# https://likegeeks.com/python-correlation-matrix/

# For data obtained from:
# https://www.kaggle.com/c/titanic/data

# Import Necessary Libraries
#import numpy as np # Linear algebra
import pandas as pd # Data processing
import os # File locating
import seaborn as sns # Constructing graphs
import matplotlib.pyplot as plt # Plotting results


### Importing Data

#train_data = r'D:\Users\Exper1mental\Clemson Classes\ME 6930\Project\titanic_train.csv'
train_csv = r'data\train.csv' # Dataset to train machine learning (ML) algorithms
test_csv = r'data\test.csv' # Dataset to test ML algorithms
answer_csv = r'data\gender_submission.csv' # Answerkey dataset

# Based on: https://stackoverflow.com/questions/35384358/how-to-open-my-files-in-data-folder-with-pandas-using-relative-path
# Find directory this python script is located in
current_file = os.path.abspath(os.path.dirname(__file__))

# 
train_path = os.path.join(current_file, train_csv)


# Create Pandas Data Frame
train_df = pd.read_csv(train_csv, index_col=0)
#print(df.columns)
#print(df.shape)

# Checking for missing data
print(train_df.isnull().sum())

# Create Heatmap of Entries Missing Data
sns.heatmap(train_df.isnull())
plt.tight_layout()
plt.show()


### Data Cleanup
# Removing unusable columns
#df_small = df.iloc[:,:6] # taking all rows but only 6 columns
train_df_s = train_df.copy()
train_df_s.drop(['name','ticket','cabin'],inplace=True,axis=1)

# Rename columns
train_df_s.rename({'home.dest' : 'Home / Destination', 'sex' : 'male'}, inplace=True, axis=1)

# Convert male into usable binary data
train_df_s.male = train_df_s.male.apply(lambda x : int(x == 'male'))
# 1 = Male; 0 = Female

# Create Matrix of Correlations
correlation_mat = train_df_s.corr()

# Create Heatmap of Correlations
sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of Titanic roster data")
plt.xlabel("passenger features")
plt.ylabel("passenger features")
plt.tight_layout()
plt.show()