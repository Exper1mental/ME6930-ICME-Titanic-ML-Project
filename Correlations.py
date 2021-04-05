### About

# Thomas Delvaux and Anish Chaluvadi
# ME-6930 036
# 04/06/2021

# Correlation code based on:
# https://likegeeks.com/python-correlation-matrix/

# For data obtained from:
# https://www.kaggle.com/c/titanic/data


### Import Necessary Libraries

import numpy as np # Linear algebra
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

# Full data filepaths
train_path = os.path.join(current_file, train_csv)
test_path = os.path.join(current_file, test_csv)
answer_path = os.path.join(current_file, answer_csv)


# Create Pandas Data Frames
train_df = pd.read_csv(train_csv, index_col=0)
test_df = pd.read_csv(train_csv, index_col=0)
answer_df = pd.read_csv(train_csv, index_col=0)

combine = [train_df, test_df] # Useful for filling in empty entries
#print(df.columns)
#print(df.shape)

# Checking for missing data
print(train_df.isnull().sum())

# Create Heatmap of Entries Missing Data
# (uncomment the below lines to obtain the plot)
#sns.heatmap(train_df.isnull())
#plt.tight_layout()
#plt.show()


### Data Cleanup

for dataset in combine: # Perform this action for both the testing and training datasets
    #  Removing unused columns
    dataset.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)


    ## Making sex data usable

    # Rename column
    dataset.rename({'Sex' : 'Male'}, inplace=True, axis=1)
    
    # Convert male column into usable binary data
    dataset['Male'] = dataset['Male'].map( {'female': 0, 'male': 1} ).astype(int)


## Age

# Method 1 based on:
# https://www.kaggle.com/startupsci/titanic-data-science-solutions

# Plot relationship between gender, age, and pclass
# (uncomment the below lines to obtain the plot)
#grid = sns.FacetGrid(train_df, row='Pclass', col='Male', height=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend()
#plt.tight_layout()
#plt.show()

# Fill in missing ages
guess_ages = np.zeros((2,3)) # Matrix to fill
for dataset in combine: # Perform this action for both the testing and training datasets
    for i in range(0, 2): # Loop through sexes
        for j in range(0, 3): # Loop through passenger classes
            # Ignoring empty entries, obtain a list of ages for the specified sex and passenger class
            guess_df = dataset[(dataset['Male'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # Guess the median age for the specific sex and passenger class
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2): # Loop through sexes
        for j in range(0, 3): # Loop through passenger classes
            # Fill in missing ages based on the passenger's sex and class
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Male == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    # Make sure all entries in Age are now integers
    dataset['Age'] = dataset['Age'].astype(int)

print(train_df.head())

# Checking for missing data
#print(train_df.isnull().sum())
#print(train_df.isnull())

# Based on: https://stackoverflow.com/questions/51374068/how-to-remove-a-row-which-has-empty-column-in-a-dataframe-using-pandas
# Removed any rows missing data. Should only remove 2 rows missing embarked data.
train_df = train_df.dropna()

# Confirm there are no more entries missing data
#print(train_df.isnull().sum())


#sns.boxplot(x='SibSp',y='Age',data=train_df)
#plt.tight_layout()
#plt.show()
# FILL IN OR REMOVE

# Create Heatmap to Check Again for Entries Missing Data
# (uncomment the below lines to obtain the plot)
#sns.heatmap(train_df.isnull())
#plt.tight_layout()
#plt.show()

# Plot should show no missing entries, indicating the data is ready for use with ML


## Correlations

# Create Matrix of Correlations
correlation_mat = train_df.corr()

# Create Heatmap of Correlations
sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of Titanic roster data")
plt.xlabel("passenger features")
plt.ylabel("passenger features")
plt.tight_layout()
plt.show()


# Thomas's ML Algorithms
# 1) Decision Trees


# 2) Random Forest

