import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('credit_card_fraud_dataset.csv')

# Data Exploration
print("Dataset Overview:")
print(f"Total transactions: {len(df)}")
print(f"Fraudulent transactions: {df['IsFraud'].sum()} ({df['IsFraud'].mean()*100:.2f}%)")
print("\nFirst 5 transactions:")
print(df.head())

# Feature Engineering
def preprocess_data(df):
    # Convert transaction date to datetime and extract features
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['Hour'] = df['TransactionDate'].dt.hour
    df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
    df['DayOfMonth'] = df['TransactionDate'].dt.day
    
    # Encode categorical variables
    le = LabelEncoder()
    df['TransactionType'] = le.fit_transform(df['TransactionType'])
    df['Location'] = le.fit_transform(df['Location'])
    
    # Drop original date and ID columns
    df = df.drop(['TransactionID', 'TransactionDate', 'MerchantID'], axis=1)
    
    return df

df_processed = preprocess_data(df.copy())

# Split data into features and target
X = df_processed.drop('IsFraud', axis=1)
y = df_processed['IsFraud']

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Create and train the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model for later use
joblib.dump(pipeline, 'credit_card_fraud_model.pkl')

# Fraud Detection System
class FraudDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = StandardScaler()
        
    def preprocess_transaction(self, transaction):
        # Convert to DataFrame if single transaction
        if isinstance(transaction, dict):
            transaction = pd.DataFrame([transaction])
        
        # Feature engineering (same as training)
        transaction['TransactionDate'] = pd.to_datetime(transaction['TransactionDate'])
        transaction['Hour'] = transaction['TransactionDate'].dt.hour
        transaction['DayOfWeek'] = transaction['TransactionDate'].dt.dayofweek
        transaction['DayOfMonth'] = transaction['TransactionDate'].dt.day
        
        # Encode categorical variables
        le = LabelEncoder()
        transaction['TransactionType'] = le.fit_transform(transaction['TransactionType'])
        transaction['Location'] = le.fit_transform(transaction['Location'])
        
        # Drop unused columns
        transaction = transaction.drop(['TransactionID', 'TransactionDate', 'MerchantID'], axis=1, errors='ignore')
        
        return transaction
    
    def detect_fraud(self, transaction):
        processed = self.preprocess_transaction(transaction)
        proba = self.model.predict_proba(processed)[0]
        prediction = self.model.predict(processed)[0]
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': float(proba[1]),
            'transaction_details': transaction
        }

# Sample Usage
print("\nSample Fraud Detection:")

# Load the trained model
detector = FraudDetector('credit_card_fraud_model.pkl')

# Sample transactions to test
sample_transactions = [
    {
        "TransactionID": 2001,
        "TransactionDate": "2024-09-15 03:14:35.462794",
        "Amount": 4500.00,
        "MerchantID": 123,
        "TransactionType": "purchase",
        "Location": "San Jose"
    },
    {
        "TransactionID": 2002,
        "TransactionDate": "2024-09-15 12:30:35.462794",
        "Amount": 120.50,
        "MerchantID": 456,
        "TransactionType": "purchase",
        "Location": "Chicago"
    },
    {
        "TransactionID": 2003,
        "TransactionDate": "2024-09-15 23:45:35.462794",
        "Amount": 3200.00,
        "MerchantID": 789,
        "TransactionType": "refund",
        "Location": "New York"
    }
]

# Detect fraud for each sample transaction
for i, transaction in enumerate(sample_transactions, 1):
    result = detector.detect_fraud(transaction)
    print(f"\nTransaction {i} Analysis:")
    print(f"Amount: ${transaction['Amount']:.2f}")
    print(f"Type: {transaction['TransactionType'].title()}")
    print(f"Location: {transaction['Location']}")
    print(f"Fraud Prediction: {'Fraudulent' if result['is_fraud'] else 'Legitimate'}")
    print(f"Fraud Probability: {result['fraud_probability']*100:.2f}%")
    print("="*50)