
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def load_data(path='creditcard.csv'):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    return X_res, y_res

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def train_models(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    rf_cv = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc')
    rf_cv.fit(X_train, y_train)

    svm = SVC(probability=True, random_state=42)
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    svm_cv = GridSearchCV(svm, svm_params, cv=3, scoring='roc_auc')
    svm_cv.fit(X_train, y_train)

    return rf_cv.best_estimator_, svm_cv.best_estimator_

def evaluate(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.4f}\n")
    plt.figure()
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

def main():
    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Training models...")
    rf_model, svm_model = train_models(X_train, y_train)
    models = {'Random Forest': rf_model, 'SVM': svm_model}
    print("Evaluating models...")
    evaluate(models, X_test, y_test)

if __name__ == "__main__":
    main()
