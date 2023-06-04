import pandas as pd

df = pd.read_csv('TestingResultsMulti.csv', header=None)

labels = df[128]

lines = ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
count = 0
for a in range(0, 5):
    for b in range(0, 20):
        lines[b] += f"& {labels[count]} &  "

        count += 1

for i in range(len(lines)):
    print("", str(lines[i])[:-4], "\\\\")
