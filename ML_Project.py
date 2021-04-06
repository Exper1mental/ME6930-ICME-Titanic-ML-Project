########################################################################
##### About

# Authors: Anish Chaluvadi and Thomas Delvaux
# ME-6930 036
# 04/06/2021

# For data obtained from:
# https://www.kaggle.com/c/titanic/data

show_figures = 1 # all figures are displayed in the code
# 0 ~ false; 1 ~ true

print_extra_info = 1 # prints extra intermediary information
# as the script runs.

########################################################################
##### Thomas's Data Cleaning and Correlations
### Import Necessary Libraries

import numpy as np # Linear algebra
import pandas as pd # Data processing
import os # File locating
import seaborn as sns # Constructing graphs
import matplotlib.pyplot as plt # Plotting results
import random as rnd # Random number generator


### Importing Data

#train_data = r'D:\Users\Exper1mental\Clemson Classes\ME 6930\Project\titanic_train.csv'
train_csv = r'data\train.csv' # Dataset to train machine learning (ML) algorithms
test_csv = r'data\test.csv' # Dataset to test ML algorithms
answer_csv = r'data\gender_submission.csv' # Answerkey dataset
combined_csv = r'data\test_train_combined.csv' # Dataset with merging solutions and training data into
# one big dataset

# Based on: https://stackoverflow.com/questions/35384358/how-to-open-my-files-in-data-folder-with-pandas-using-relative-path
# Find directory this python script is located in
#current_file = os.path.abspath(os.path.dirname(__file__))

# Full data filepaths
#train_path = os.path.join(current_file, train_csv)
#test_path = os.path.join(current_file, test_csv)
#answer_path = os.path.join(current_file, answer_csv)


# Create Pandas Data Frames
train_df = pd.read_csv(train_csv, index_col=0)
test_df = pd.read_csv(test_csv, index_col=0)
answer_df = pd.read_csv(answer_csv, index_col=0)
combined_df = pd.read_csv(combined_csv, index_col=0)

combine = [combined_df] #[train_df, test_df] # Useful for filling in empty entries
#print(df.columns)
#print(df.shape)

# Checking for missing data
#print(combined_df.isnull().sum())

# Create Heatmap of Entries Missing Data
# (uncomment the below lines to obtain the plot)
if show_figures == 1:
    sns.heatmap(combined_df.isnull())
    plt.title('Heatmap of Uncleaned Data (Empty Entries in White)')
    plt.tight_layout()
    plt.show()

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
if show_figures == 1:
    grid = sns.FacetGrid(combined_df, row='Pclass', col='Male', height=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()
    plt.tight_layout()
    plt.show()

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

# Preview the data
#print(combined_df.head())

# Checking for missing data
#print(combined_df.isnull().sum())
#print(combined_df.isnull())

# Based on: https://stackoverflow.com/questions/51374068/how-to-remove-a-row-which-has-empty-column-in-a-dataframe-using-pandas
# Removed any rows missing data. Should only remove 2 rows missing embarked data.
combined_df = combined_df.dropna()

## Making embarkation data usable
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Confirm there are no more entries missing data
#print(combined_df.isnull().sum())

### Feature Engineering

## Title
for dataset in combine: # Perform this action for both the testing and training datasets
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') # These are rare titles which may indicate a different relevance than other titles

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} # Ensure it is numerical for ML purposes
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
## Family Size
for dataset in combine: # Perform this action for both the testing and training datasets
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 # Creates a separate single variable for family size

## Split Embarked data into three boolean columns

combined_df = pd.get_dummies(combined_df)
# Split the string-based embarked data into three columns with boolean 
# integers (in other words, convert the embarked data into a form that
# Random Forest can use)

# Most algorithms, including Random Forest and Decision Trees can't use
# columns with string-based data, making this conversion necessary.

# Preview New Data Format
#print(combined_df.head())


# Create Heatmap to Check Again for Entries Missing Data
# (uncomment the below lines to obtain the plot)
if show_figures == 1:
    sns.heatmap(combined_df.isnull())
    plt.tight_layout()
    plt.show()
    # The heatmap should be a big red/pink square,
    # which indicates that no empty entries are present
    #
    # If it is deep purple / navy blue, there are empty
    # entries that somehow were missed. This should not
    # happen.

# Plot should show no missing entries, indicating the data is ready for use with ML

### Correlations

# Correlation code based on:
# https://likegeeks.com/python-correlation-matrix/

# Create Matrix of Correlations
correlation_mat = combined_df.corr()

# Create Heatmap of Correlations
if show_figures == 1:
    sns.heatmap(correlation_mat, annot = True)
    plt.title("Correlation matrix of Titanic roster data")
    plt.xlabel("passenger features")
    plt.ylabel("passenger features")
    plt.tight_layout()
    plt.show()


########################################################################
##### Thomas's ML Algorithms
# Based on: https://www.kaggle.com/vanshjatana/applied-machine-learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ML Algorithm 1: Decision Trees
from sklearn.tree import DecisionTreeClassifier
#data = pd.read_csv('../input/classification-suv-dataset/Social_Network_Ads.csv')
dt = combined_df.copy()
#dt.head()

# X contains the data, Y contains the "solution"
# We drop the solution column "Survived" out of the data for testing and training.

# Column "Embarked" must also be dropped because it contained non-float data, which
# is incompatible with the Decision Trees algorithm
X = dt.drop(['Survived','Embarked_S','Embarked_C','Embarked_Q'], axis=1)
#X = dt.drop(['Survived','Embarked'], axis=1) #dt.iloc[:, [2,3]].values
y = dt['Survived'] #dt.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
acc_dt=accuracy_score(y_test, y_pred)
print(f'Decision Trees Accuracy: {round(acc_dt*100,3)}%')

# ML Algorithm 2: Random Forest
from sklearn.ensemble import RandomForestClassifier
#rf = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
rf = combined_df.copy()
#rf.head()

X = rf.drop('Survived', axis=1)
y = rf['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)
acc_rf = model.score(X_test, y_test)
print(f'Random Forest Accuracy: {round(acc_rf*100,3)}%')

########################################################################
##### Anish's ML Algorithms
# Based on: https://www.kaggle.com/vinothan/titanic-model-with-90-accuracy; https://www.kaggle.com/startupsci/titanic-data-science-solutions#Titanic-Data-Science-Solutions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ML Algorithm 3: Linear Regression
from sklearn.linear_model import LogisticRegression

lr = combined_df.copy
X_train_lr = lr.drop('Survived', axis=1)
Y_train_lr = lr['Survived']

logreg = LogisticRegression()
logreg.fit(X_train_lr, Y_train_lr)
Y_pred_lr = logreg.predict(X_test)
acc_log = logreg.score(X_train_lr, Y_train_lr)
print(f'Logistic Regression Accuracy: {round(acc_log*100,3)}%')

coeff_df = pd.DataFrame(train_df.columns.delete(0)) # Details correlations between features for better understanding (I think we should try to do this for all of our models)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)

# ML Algorithm 4: k-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knndf = combined_df.copy
X_train_knn = knndf.drop('Survived', axis=1)
Y_train_knn = knndf['Survived']

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_knn, Y_train_knn)
Y_pred_knn = knn.predict(X_test)
acc_knn = knn.score(X_train_knn, Y_train_knn)
print(f'k-Nearest Neighbors Accuracy: {round(acc_knn*100,3)}%')

### Post-processing

# Model Evaluation
models = pd.DataFrame({
    'Model': ['Decision Trees', 'Random Forest', 'Logistic Regression', 'k-Nearest Neighbors'],
    'Score': [acc_dt, acc_rf, acc_log, acc_knn]})
models.sort_values(by='Score', ascending=False)
