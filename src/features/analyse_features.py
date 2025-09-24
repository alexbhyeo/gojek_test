import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from src.utils.store import AssignmentStore

# Create sample data
store = AssignmentStore()
#data_path = "/Users/yeoboonhong/Documents/gojek_assignment/ds-assignment-master/data/processed/transformed_dataset.csv"
data = store.get_processed("transformed_dataset.csv")
#data = pd.read_csv(data_path)
X = data.drop(["is_completed", "event_timestamp", "participant_status", "booking_history"], axis=1)
y = data["is_completed"]
feature_names = X.columns
print(feature_names)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=33)
model.fit(X, y)

# Get feature importance
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
# plt.show()
plt.savefig('feature_important.png')
