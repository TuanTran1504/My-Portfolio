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
print(data.head())

#Data subset
data_sample = data.sample(frac=0.001, random_state=42)
#Check for missing Value

missing_values = data_sample.isnull().sum()
print(missing_values[missing_values > 0])

#Handling missing value
numeric_cols = data_sample.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = data_sample.select_dtypes(include=['object', 'category']).columns

data_sample[numeric_cols] = data_sample[numeric_cols].fillna(data_sample[numeric_cols].median())
data_sample[non_numeric_cols] = data_sample[non_numeric_cols].fillna(data_sample[non_numeric_cols].mode().iloc[0])

#Check the result
print(data_sample.isnull().sum())

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


# Hyperparameter tuning function
def tune_hyperparameters():
    models = [SVC(), RandomForestClassifier(), DecisionTreeClassifier()]
    kfold = KFold(n_splits=10, shuffle=True)
    best_models = {}
    best_scores = {}

    for model in models:
        if isinstance(model, SVC):
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
        elif isinstance(model, RandomForestClassifier):
            param_grid = {'n_estimators': [100, 170, 200], 'max_depth': [None, 5, 10, 15], 'min_samples_leaf': [2, 4, 6], 'max_features': ['sqrt', 'log2', None]}
        elif isinstance(model, DecisionTreeClassifier):
            param_grid = {'max_features': ['sqrt', 'log2', None], 'max_depth': [None, 3, 5, 7], 'min_samples_leaf': [1, 2, 4, 6], 'min_samples_split': [2, 5, 10]}

        grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='f1_macro')
        grid_search.fit(X_train_preprocessed, y_train)

        model_name = model.__class__.__name__
        if model_name not in best_scores or grid_search.best_score_ > best_scores[model_name]:
            best_models[model_name] = grid_search.best_estimator_
            best_scores[model_name] = grid_search.best_score_
            print(f'Best Parameters for {model_name}: {grid_search.best_params_}')
            print(f'Best Cross Validation score for {model_name}: {grid_search.best_score_:.4f}')
            print('----------------------------------------')

    return best_models

# Tune hyperparameters and get the best models
best_models = tune_hyperparameters()

def compare_model_crossvalidation(best_models):
    kfold = KFold(n_splits=10, shuffle=True)
    for model_name, model in best_models.items():
        scores = []
        for _ in range(10):  # Run 10 iterations
            cv_score = cross_val_score(model, X_train_preprocessed, y_train, cv=kfold, scoring='f1_macro')
            scores.append(cv_score.mean())
        avg_score = np.mean(scores)
        print(f'Average Cross Validation score of {model_name}: {avg_score}')

# Perform cross-validation on the best models
compare_model_crossvalidation(best_models)
