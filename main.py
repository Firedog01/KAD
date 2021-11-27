from pandas import read_csv

# Wczytywanie
reader = read_csv("data1.csv", header=None)
data = reader.values.tolist()
