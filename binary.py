import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


# Load the CSV file into a pandas DataFrame
df = pd.read_csv('TrainingDataBinary.csv', header=None)

# Label column in training data
target = 128

average = 0
high, low = None, None
for i in range (0, 100):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), df[target], test_size=(0.26)
    )

    # Create an instance of the SGDClassifier
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100))

    # Train classifier
    clf.fit(X_train, y_train)

    # Predict accuracy test
    y_pred = clf.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy {i}: {accuracy}")

    average += accuracy

    if high == None or high < accuracy:
        high = accuracy

    if low == None or low > accuracy:
        low = accuracy

average = average / 100
print(f"TOTAL: {average}")

print(f"HIGH {high} - LOW {low}")

# Classify testing data
#test = pd.read_csv('TestingDataBinary.csv', header=None)

#y_out = clf.predict(test)
#test[128] = y_out

#test.to_csv('TestingResultsBinary.csv', index=False, header=False)