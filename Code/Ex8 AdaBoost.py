# Program to implement Boosting using AdaBoost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
from sklearn.datasets import load_wine
data = load_wine()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create AdaBoost model
model = AdaBoostClassifier(
    n_estimators=50,      # Number of weak learners
    learning_rate=1.0,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Display results
print("Actual Values : ", y_test)
print("Predicted Values:", y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
