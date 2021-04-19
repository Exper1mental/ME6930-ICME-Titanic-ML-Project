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
run_graphviz = 1 # runs graphviz code to create PDF visualizations
# for the Decision Tree model

print_extra_info = 0 # prints extra intermediary information
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


### Thomas's Importing Data
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
if show_figures == 1:
    sns.heatmap(combined_df.isnull())
    plt.title('Heatmap of Uncleaned Data (Empty Entries in White)')
    plt.tight_layout()
    plt.show()

### Thomas's Data Cleanup

for dataset in combine: # Perform this action for both the testing and training datasets
    #  Removing unused columns
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
#print('Empty entries after filling in age data:')
#print(combined_df.isnull().sum())
#print(combined_df.isnull())


### Anish's Feature Engineering

## Title
   
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} # Define what titles are

for dataset in combine: # Perform this action for both the testing and training datasets
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # Extract titles from names
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') # These are rare titles which may indicate a different relevance than other titles
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)  # Convert titles into numbers for ML purposes
    dataset['Title'] = dataset['Title'].fillna(0)

combined_df = combined_df.drop(['Name'], axis=1)


## Family Size

#for dataset in combine: # Perform this action for both the testing and training datasets
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1 # Creates a separate single variable for family size
#dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 # Creates a separate single variable for family size

    ## Split Embarked data into three boolean columns
    #dataset = pd.get_dummies(dataset)


## Thomas's Remove missing data

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
# Emptry entries had to be removed before this can be done.

# OPTION 1: Change strings to integers

#ports = {"S": 0, "C": 1, "Q": 2}
#combined_df['Embarked'] = combined_df['Embarked'].map(ports)

# OPTION 2: Split Embarked data into three boolean columns

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


### Thomas's Correlations

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
print('\nThomas\'s ML Algorithm Results:')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ML Algorithm 1: Decision Trees
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Older line of code, left for compatibility
#data = pd.read_csv('../input/classification-suv-dataset/Social_Network_Ads.csv')
dt = combined_df.copy()
#dt.head()

# X contains the data, Y contains the "solution"
# We drop the solution column "Survived" out of the data for testing and training.

# Column "Embarked" must also be dropped because it contained non-float data, which
# is incompatible with the Decision Trees algorithm
X = dt.drop(['Survived'], axis=1)
y = dt['Survived'] #dt.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier=tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=100, max_depth=1, random_state=0)
classifier.fit(X_train,y_train)
y_pred_dt=classifier.predict(X_test)
acc_dt=accuracy_score(y_test, y_pred_dt)
print(f'Decision Trees Accuracy: {round(acc_dt*100,3)}%')

sns.heatmap(confusion_matrix(y_test,y_pred_dt),annot=True,fmt='3.0f',cmap="Blues")
plt.title('Decision Trees Matrix', y=1.05, size=15)
plt.savefig('Confusion_Matrix_dt.png')
plt.tight_layout()
plt.show()


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

# ML Algorithm 2: Random Forest
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

sns.heatmap(confusion_matrix(y_test,Y_pred_rf),annot=True,fmt='3.0f',cmap="Blues")
plt.title('Random Forest Confusion Matrix', y=1.05, size=15)
plt.savefig('Confusion_Matrix_rf.png')
plt.tight_layout()
plt.show()
plt.close()




########################################################################
##### Anish's ML Algorithms
print('\nAnish\'s ML Algorithm Results:')
# Based on: https://www.kaggle.com/vinothan/titanic-model-with-90-accuracy; https://www.kaggle.com/startupsci/titanic-data-science-solutions#Titanic-Data-Science-Solutions

# ML Algorithm 3: Logistical Regression
from sklearn.linear_model import LogisticRegression

lr_df = combined_df.copy()
X_train_lr = lr_df.drop(['Survived'], axis=1)
Y_train_lr = lr_df['Survived']

logreg = LogisticRegression(solver='lbfgs', max_iter=400)
logreg.fit(X_train_lr, Y_train_lr)
Y_pred_lr = logreg.predict(X_test)
acc_log = logreg.score(X_train_lr, Y_train_lr)
print(f'Logistic Regression Accuracy: {round(acc_log*100,3)}%')

sns.heatmap(confusion_matrix(y_test,Y_pred_lr),annot=True,fmt='3.0f',cmap="Blues")
plt.title('Logistical Regression Confusion Matrix', y=1.05, size=15)
plt.savefig('Confusion_Matrix_lr.png')


# Another way to view feature correlations (uncomment to see in terminal output)
# coeff_df = pd.DataFrame(combined_df.columns.delete(0)) # Details correlations between features for better understanding (I think we should try to do this for all of our models)
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
# coeff_df.sort_values(by='Correlation', ascending=False)
# #print(coeff_df)


# ML Algorithm 4: K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn_df = combined_df.copy()
X_train_knn = knn_df.drop(['Survived'], axis=1)
Y_train_knn = knn_df['Survived']

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_knn, Y_train_knn)
Y_pred_knn = knn.predict(X_test)
acc_knn = knn.score(X_train_knn, Y_train_knn)
print(f'k-Nearest Neighbors Accuracy: {round(acc_knn*100,3)}%')

sns.heatmap(confusion_matrix(y_test,Y_pred_knn),annot=True,fmt='3.0f',cmap="Blues")
plt.title('k-Nearest Neighbors Confusion Matrix', y=1.05, size=15)
plt.savefig('Confusion_Matrix_knn.png')
plt.tight_layout()
plt.show()


### Post-processing

# Model Evaluation
print('\n')
models = pd.DataFrame({
    'Model': ['Decision Trees', 'Random Forest', 'Logistic Regression', 'k-Nearest Neighbors'],
    'Score': [acc_dt, acc_rf, acc_log, acc_knn]})
models.sort_values(by='Score', ascending=False)
print(models)
