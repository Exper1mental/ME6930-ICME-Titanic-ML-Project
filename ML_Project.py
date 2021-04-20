########################################################################
##### About

# Authors: Anish Chaluvadi and Thomas Delvaux
# ME-6930 036
# 04/19/2021

# For data obtained from:
# https://www.kaggle.com/c/titanic/data

show_figures = 0 # all figures are displayed in the code
# 0 ~ false; 1 ~ true

# WARNING: Requires graphviz (both the executable file and python library)
# on Windows the executables MUST be added to PATH
# You may need to restart computer after installation before use (namely if using Windows)
run_graphviz = 0 # runs graphviz code to create PDF visualizations
# for the Decision Tree model

print_extra_info = 0 # prints extra intermediary information
# as the script runs.

# WARNING: Need to install mlxtend data visualization package for some of the plots
# Can simply be done through conda-forge or using 'pip install mlxtend' in PyPI

########################################################################
##### Data Cleaning and Correlations
### Import Necessary Libraries

import numpy as np # Linear algebra
import pandas as pd # Data processing
import os # File locating
import seaborn as sns # Constructing graphs
import matplotlib.pyplot as plt # Plotting results
import random as rnd # Random number generator


### Importing Data
train_csv = r'data\train.csv' # Dataset to train machine learning (ML) algorithms
test_csv = r'data\test.csv' # Dataset to test ML algorithms
answer_csv = r'data\gender_submission.csv' # Answerkey dataset
combined_csv = r'data\test_train_combined.csv' # Dataset with merging solutions and training data into
# one big dataset

# Create Pandas Data Frames
combined_df = pd.read_csv(combined_csv, index_col=0)

combine = [combined_df] #[train_df, test_df] # Useful for filling in empty entries
#print(df.columns)
#print(df.shape)

# Checking for missing data
if print_extra_info == 1:
    print('Empty entries before filling in age data:')
    print(combined_df.isnull().sum())

# Create Heatmap of Entries Missing Data
# (uncomment the below lines to obtain the plot)
sns.heatmap(combined_df.isnull())
plt.title('Heatmap of Uncleaned Data (Empty Entries in White)')
plt.tight_layout()
plt.savefig('MissingDataCheck_Uncleaned.png')
if show_figures == 1:
    plt.show()
plt.close()

### Data Cleanup

for dataset in combine: # Perform this action for both the testing and training datasets
    #  Removing unused columns because too much missing data
    dataset.drop(['Ticket','Cabin'],inplace=True,axis=1)


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
grid = sns.FacetGrid(combined_df, row='Pclass', col='Male', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.tight_layout()
plt.savefig('AgeDistribution.png')
if show_figures == 1:
    plt.show()
plt.close()

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
#print('Empty entries after filling in age data:')
#print(combined_df.isnull().sum())
#print(combined_df.isnull())


### Feature Engineering

## Title

import re # Creating a function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name) # If the title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in combine: # Perform this action for both the testing and training datasets
    dataset['Title'] = dataset['Name'].apply(get_title) # Use the function

for dataset in combine: # Perform this action for both the testing and training datasets
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') # Group all non-common titles into one single grouping "Rare"
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') # Group Mlle and Ms into a single category "Miss"
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs') # Group Mme with "Mrs" category
    dataset.drop(['Name'],inplace=True,axis=1) # Removing unused column now that Title feature has been designed

## Family Size

combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1 # Creates a separate single variable for family size

# Create Heatmap to Check Again for Entries Missing Data
# (uncomment the below lines to obtain the plot)
sns.heatmap(combined_df.isnull())
plt.tight_layout()
plt.savefig('MissingDataCheck_PartiallyCleaned.png')
if show_figures == 1:
    plt.show()
plt.close()
    # The heatmap should be a big red/pink square,
    # which indicates that no empty entries are present
    #
    # If it is deep purple / navy blue, there are empty
    # entries that somehow were missed. This should not
    # happen.

# Checking for missing data
if print_extra_info == 1:
    print('\nEmpty entries before removing missing data:')
    print(combined_df.isnull().sum())

# Remove missing dats
# Based on: https://stackoverflow.com/questions/51374068/how-to-remove-a-row-which-has-empty-column-in-a-dataframe-using-pandas
# Removed any rows missing data. Should only remove 2 rows missing embarked data.
combined_df = combined_df.dropna()

# Confirm there are no more entries missing data
#print(combined_df.isnull().sum())


## Making embarkation data usable
# Empty entries had to be removed before this can be done.

# OPTION 1: Change strings to integers

#ports = {"S": 0, "C": 1, "Q": 2}
#combined_df['Embarked'] = combined_df['Embarked'].map(ports)

# OPTION 2: Split categorical data into separate boolean columns

combined_df = pd.get_dummies(combined_df)
# Split the string-based embarked data into three columns with boolean 
# integers (in other words, convert the embarked data into a form that
# Random Forest can use)

# Most algorithms, including Random Forest and Decision Trees can't use
# columns with string-based data, making this conversion necessary.

# Preview New Data Format
#print(combined_df.head())

# Checking for missing data
if print_extra_info == 1:
    print('\nEmpty entries after removing missing data:')
    print(combined_df.isnull().sum())


# Create Heatmap to Check Again for Entries Missing Data
# (uncomment the below lines to obtain the plot)
sns.heatmap(combined_df.isnull())
plt.tight_layout()
plt.savefig('MissingDataCheck_Cleaned.png')
if show_figures == 1:
    plt.show()
plt.close()


    
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

# Create Matrix of Correlations for Training Dataset
from sklearn.model_selection import train_test_split
corr = combined_df.copy()
X = corr.copy()
#X = dt.drop(['Survived','Embarked'], axis=1) #dt.iloc[:, [2,3]].values
y = corr['Survived'] #dt.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
correlation_mat = X_train.corr()
#correlation_mat = combined_df.corr()

# Create Heatmap of Correlations

sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of Titanic roster data")
plt.xlabel("passenger features")
plt.ylabel("passenger features")
plt.tight_layout()
plt.savefig('Correlations.png')
if show_figures == 1:
    plt.show()
plt.close()

########################################################################
##### Thomas's ML Algorithms
# Based on: https://www.kaggle.com/vanshjatana/applied-machine-learning
print('\nThomas\'s ML Algorithm Results:')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

## ML Algorithm 1: Decision Trees
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Older line of code, left for compatibility
#data = pd.read_csv('../input/classification-suv-dataset/Social_Network_Ads.csv')
dt = combined_df.copy()
#dt.head()

# X contains the data, Y contains the "solution"
# We drop the solution column "Survived" out of the data for testing and training.

# Column "Embarked" must also be dropped because it contained non-float data, which
# is incompatible with the Decision Trees algorithm
X = dt.drop(['Survived'], axis=1) # Include the whole dataset as features outside of the Survived
y = dt['Survived'] #dt.iloc[:, 4].values # Match the X with the Survived column from the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # Splitting combined dataset into 80% train and 20% test
sc_X = StandardScaler() # Scale and normalize the features into normal distribution
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=100, max_depth=1, random_state=0) # Calling the decision tree classifier from sklearn
classifier.fit(X_train,y_train)
y_pred_dt=classifier.predict(X_test)
acc_dt=accuracy_score(y_test, y_pred_dt) # Calculate the accuracy of decision tree algorithm
print(f'Decision Trees Accuracy: {round(acc_dt*100,3)}%')

# Confusion Matrix
# Based on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
sns.heatmap(confusion_matrix(y_test,y_pred_dt),annot=True,fmt='3.0f',cmap="Blues") # Plotting confusion matrix and cross-validation value for the decision tree algorithm
plt.title('Decision Trees Matrix', y=1.05, size=15)
plt.savefig('Confusion_Matrix_dt.png')
plt.tight_layout()
if show_figures==1:
    plt.show()
plt.close()

# Documentation on decision trees:
# https://scikit-learn.org/stable/modules/tree.html#tree


# Generate decision tree visualization PDF
# WARNING: Requires graphviz (both the executable file and python library)
# You may need to restart computer after installation before use

# Helpful for getting class names from model:
# https://stackoverflow.com/questions/39476020/get-feature-and-class-names-into-decision-tree-using-export-graphviz
#print(classifier.classes_.astype(str))
#print(['Died', 'Survived'])

# Documentation on exporting to graphviz:
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
if run_graphviz == 1:
    import graphviz 
    dot_data = tree.export_graphviz(classifier, out_file=None,
                        feature_names=X.columns,  
                        # Use 'Died' for '0' and 'Survived' for '1' in class name
                        class_names=['Died', 'Survived'], #classifier.classes_.astype(str),
                        filled=True, rounded=True,  
                        special_characters=True,
                        proportion=True) 
    graph = graphviz.Source(dot_data)
    graph.render("titanic_decision-tree")


#print(y_train.astype(str).loc[:])

# Feature Importance for Decision Tree
# Loosely based on https://machinelearningmastery.com/calculate-feature-importance-with-python/ but mainly from Anish's previous research experience
# For more comments on each line of code, see the feature importance section for random forest algorithm
importance = [] # Create an array called importance
names      = [] # Create an array called names

features = [r'Pclass', r'Male', r'Age', r'SibSp', r'Parch', r'Fare', r'Family Size', r'Embarked_C', r'Embarked_Q', r'Embarked_S', r'Title_Master', r'Title_Miss', r'Title_Mr', r'Title_Mrs', r'Title_Rare'] # List all selected features (to be used as x-axis labels in the plot)

for i in range(len(classifier.feature_importances_)):
    if classifier.feature_importances_[i] > 0.005: # 0.005 is a general alpha threshold value chosen to determine an "important" feature
        importance.append(classifier.feature_importances_[i]) # Add importance value to importance array
        names.append(features[i]) # Add feature name to names array

#print(importance)
#print(names)
#print(len(names))

fig, ax = plt.subplots(figsize=(14, 6))
y_pos     = np.arange(len(names))
bar_width = 0.20
opacity   = 0.5

plt.barh(y_pos + 0*bar_width, importance, alpha=opacity, color='b', label='xxx')

plt.yticks(y_pos, names)
plt.tick_params(axis='x', labelsize = 15)
plt.tick_params(axis='y', labelsize = 15)
plt.grid(False)
plt.ylabel('Features', fontsize = 20)
plt.tight_layout()
fig.savefig('feature_importance_dt.png', bbox_inches='tight', dpi=400);
if show_figures==1:
    plt.show()
plt.close()


## ML Algorithm 2: Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = combined_df.copy()
#rf.head()

X = rf.drop(['Survived'], axis=1)
y = rf['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, max_depth=6, random_state=0)
model.fit(X_train, y_train)
Y_pred_rf = model.predict(X_test)
acc_rf = model.score(X_test, y_test)
print(f'Random Forest Accuracy: {round(acc_rf*100,3)}%')

# Confusion Matrix
# Based from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
sns.heatmap(confusion_matrix(y_test,Y_pred_rf),annot=True,fmt='3.0f',cmap="Blues")
plt.title('Random Forest Confusion Matrix', y=1.05, size=15)
plt.tight_layout()
plt.savefig('Confusion_Matrix_rf.png')
if show_figures==1:
    plt.show()
plt.close()

# Feature Importance for Random Forest
# Loosely based on https://machinelearningmastery.com/calculate-feature-importance-with-python/ but mainly from Anish's previous research experience
importance = [] # Create an array called importance
names      = [] # Create an array called names

features = [r'Pclass', r'Male', r'Age', r'SibSp', r'Parch', r'Fare', r'Family Size', r'Embarked_C', r'Embarked_Q', r'Embarked_S', r'Title_Master', r'Title_Miss', r'Title_Mr', r'Title_Mrs', r'Title_Rare'] # List all selected features (to be used as x-axis labels in the plot)

for i in range(len(model.feature_importances_)):
    if model.feature_importances_[i] > 0.005: # 0.005 is a general alpha threshold value chosen to determine an "important" feature
        importance.append(model.feature_importances_[i]) # Add importance value to importance array
        names.append(features[i]) # Add feature name to names array

#print(importance)
#print(names)
#print(len(names))

fig, ax = plt.subplots(figsize=(14, 6))

y_pos     = np.arange(len(names))
bar_width = 0.20
opacity   = 0.5

plt.barh(y_pos + 0*bar_width, importance, alpha=opacity, color='b', label='xxx')

plt.yticks(y_pos, names)
plt.tick_params(axis='x', labelsize = 15)
plt.tick_params(axis='y', labelsize = 15)
plt.grid(False)
plt.ylabel('Features', fontsize = 20)
plt.tight_layout()
fig.savefig('feature_importance_rf.png', bbox_inches='tight', dpi=400);
if show_figures==1:
    plt.show()
plt.close()

########################################################################
##### Anish's ML Algorithms
print('\nAnish\'s ML Algorithm Results:')
# Based on: https://www.kaggle.com/vinothan/titanic-model-with-90-accuracy; https://www.kaggle.com/startupsci/titanic-data-science-solutions#Titanic-Data-Science-Solutions

## ML Algorithm 3: Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_df = combined_df.copy() # Create a copy of the dataframe for Logistic Regression algorithm

X = lr_df.drop(['Survived'], axis=1) # Establish all the features to be used in the algorithm (i.e. drop the survived column)
y = lr_df['Survived'] # Needs to model only with respect to the survived column
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size = 0.20, random_state = 0)

logreg = LogisticRegression(solver='lbfgs', max_iter=1000) # Calling the logreg function from sklearn
logreg.fit(X_train_lr, y_train_lr) # Determining best fit of the logistic regression from training data (which is the same 80% of passengers as above)
Y_pred_lr = logreg.predict(X_test_lr) # Predicting Y values depending on test set
acc_log = logreg.score(X_test_lr, y_test_lr) # Calculating accuracy
print(f'Logistic Regression Accuracy: {round(acc_log*100,3)}%')

# Confusion Matrix
# Based from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
sns.heatmap(confusion_matrix(y_test,Y_pred_lr),annot=True,fmt='3.0f',cmap="Blues")
plt.title('Logistical Regression Confusion Matrix', y=1.05, size=15)
plt.tight_layout()
plt.savefig('Confusion_Matrix_lr.png')
if show_figures==1:
    plt.show()
plt.close()

# Data Analysis Visualization

print(logreg.intercept_) # Gives the intercept of the best fit logistic regression equation found by the algorithm
print(logreg.coef_) # Gives the coefficients of the best fit logistic regression equation for each feature found by the algorithm

plt.plot(X_test, Y_pred_lr, 'o'); # Plotting the data points predicted
plt.tight_layout()
plt.savefig('Plot_lr.png')
if show_figures==1:
    plt.show()
plt.close()

# Feature Importance for Logistic Regression

# Based on: https://machinelearningmastery.com/calculate-feature-importance-with-python/ but mainly from Anish's previous research experience

importance = logreg.coef_[0] # Get feature importance values from algorithm
plt.bar([r'Pclass', r'Male', r'Age', r'SibSp', r'Parch', r'Fare', r'Family Size', r'Embarked_C', r'Embarked_Q', r'Embarked_S', r'Title_Master', r'Title_Miss', r'Title_Mr', r'Title_Mrs', r'Title_Rare'], importance) # Plot feature importance for logistic regression algorithm

plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize = 12)
plt.tick_params(axis='y', labelsize = 15)
plt.grid(False)
plt.xlabel('Features', fontsize = 20)
plt.tight_layout()

fig.savefig('feature_importance_lr.png', bbox_inches='tight', dpi=400);
if show_figures==1:
    plt.show()
plt.close()

## ML Algorithm 4: K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn_df = combined_df.copy()
X = knn_df.drop(['Survived'], axis=1)
y = knn_df['Survived']
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size = 0.20, random_state = 0)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_knn, y_train_knn)
Y_pred_knn = knn.predict(X_test_knn)
acc_knn = knn.score(X_test_knn, y_test_knn)
print(f'k-Nearest Neighbors Accuracy: {round(acc_knn*100,3)}%')

# Confusion Matrix
# Based from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
sns.heatmap(confusion_matrix(y_test,Y_pred_knn),annot=True,fmt='3.0f',cmap="Blues")
plt.title('k-Nearest Neighbors Confusion Matrix', y=1.05, size=15)
plt.tight_layout()
plt.savefig('Confusion_Matrix_knn.png')
if show_figures==1:
    plt.show()
plt.close()

# Feature Importance for k-Nearest Neighbors

# Based on https://machinelearningmastery.com/calculate-feature-importance-with-python/
# k-Nearest Neighbors is one of the algorithms that does not support feature importance or feature selection natively

##################################################################################### Post-processing

## Model Evaluation
print('\n')
models = pd.DataFrame({
    'Model': ['Decision Trees', 'Random Forest', 'Logistic Regression', 'k-Nearest Neighbors'],
    'Score': [acc_dt, acc_rf, acc_log, acc_knn]}) # Tag each ML algorithm with its respective accuracy score
models.sort_values(by='Score', ascending=False) # Sort by descending score
print(models)
