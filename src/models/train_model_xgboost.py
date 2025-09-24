import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore

def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    feature_columns = config["features"]

    X = df[feature_columns]
    y = df[config["target"]]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=33, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=33
    )

    # Train the model
    xgb_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test_scaled)
    #y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate the model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)

        # Extract values
    tn, fp, fn, tp = cm.ravel()
    print(tn, fp, fn, tp)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("precision : ", precision)
    print("recall : ", recall)

    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
