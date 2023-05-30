import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load the CSV file into a pandas DataFrame
df = pd.read_csv('TrainingDataMulti.csv', header=None)

# Label column in training data
target = 128

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis=1), df[target], test_size=(0.36), random_state=42
)

# Create an instance of the SGDClassifier
clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-2, loss='log_loss', random_state=42))

# Train classifier
clf.fit(X_train, y_train)

# Predict accuracy test
y_pred = clf.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Classify testing data
test = pd.read_csv('TestingDataMulti.csv', header=None)

y_out = clf.predict(test)
test[128] = y_out

test.to_csv('TestingResultsMulti.csv', index=False, header=False)