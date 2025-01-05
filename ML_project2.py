import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = 'dataset ml project.csv'  # Replace with the correct path but here only name of the file is written because the file location is same as the code location
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())
print("\n")

# Display the summary of the dataset
print(data.info())
print("\n")

# Check for any missing values
print(data.isnull().sum())
print("\n")

# Encode the 'type' categorical variable
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

# Define feature columns and target variable
feature_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
target_column = 'isFraud'

# Standardize the numerical features
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Create new feature: difference in balance before and after the transaction for both origin and destination accounts
data['balanceOrigDiff'] = data['newbalanceOrig'] - data['oldbalanceOrg']
data['balanceDestDiff'] = data['newbalanceDest'] - data['oldbalanceDest']

# Include the new features in the feature set
feature_columns = feature_columns + ['balanceOrigDiff', 'balanceDestDiff']


# Split the data manually
# X_train = data.loc[:2, feature_columns]
# y_train = data.loc[:2, target_column]
# X_test = data.loc[3:4, feature_columns]
# y_test = data.loc[3:4, target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data[target_column], test_size=0.2, random_state=42, stratify=None)

# X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data[target_column], train_size=4, test_size=1, random_state=42, stratify=None)


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the first 3 rows
rf_classifier.fit(X_train, y_train)

# Predict the target variable for the next 2 rows
predictions = rf_classifier.predict(X_test)

# Print the predictions
print("Predictions:", predictions)
print("Actual values:", y_test.values)
print("Confusion Matrix:\n", confusion_matrix(y_test.values,  predictions, labels=[0,1]))
print("Accuracy Score:\n", accuracy_score(y_test.values, predictions))
print("Classification Report:\n", classification_report(y_test.values,  predictions))
