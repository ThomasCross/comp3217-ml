import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Load the CSV file into a pandas DataFrame
df = pd.read_csv('TrainingDataBinary.csv', header=None)

target = 128

for a in range(1, 100):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), df[target], test_size=(0.26)
    )

    # Create an instance of the SGDClassifier
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, loss='log_loss'))

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

# Get testing data classified
test = pd.read_csv('TestingDataBinary.csv', header=None)

y_out = clf.predict(test)
test[128] = y_out

test.to_csv('TestingResultsBinary.csv', index=False, header=False)