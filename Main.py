import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import numpy as np
# Load the dataset
file_path = '~/Documents/Python/Test/Portfolio/archive/lending_club_loan_two.csv'
data = pd.read_csv(file_path)

#Data subset
data_sample = data.sample(frac=1, random_state=42)
#Check for missing Value

missing_values = data_sample.isnull().sum()


#Handling missing value
numeric_cols = data_sample.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = data_sample.select_dtypes(include=['object', 'category']).columns

data_sample[numeric_cols] = data_sample[numeric_cols].fillna(data_sample[numeric_cols].median())
data_sample[non_numeric_cols] = data_sample[non_numeric_cols].fillna(data_sample[non_numeric_cols].mode().iloc[0])

#Prediction Target
data_sample['loan_status'] = data_sample['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

#Remove loan_status from the non_numerical_cols
numeric_cols = list(numeric_cols)
non_numeric_cols = list(non_numeric_cols)
non_numeric_cols.remove('loan_status')
non_numeric_cols.remove('address')


#Preprocessing data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), non_numeric_cols)
    ])

X = data_sample[numeric_cols + non_numeric_cols]
y = data_sample['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

dtc=DecisionTreeClassifier(max_depth=None,max_features=None,min_samples_split=5,min_samples_leaf=2)
dtc.fit(X_train_preprocessed,y_train)
pred=dtc.predict(X_test_preprocessed)
accuracy = accuracy_score(y_test, pred)
print(accuracy)