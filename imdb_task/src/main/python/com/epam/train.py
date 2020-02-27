from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import pandas
from sklearn.model_selection import train_test_split

input_file = "../../../resources/IMDB Dataset.csv"
data = pandas.read_csv(input_file, header = 0)
original_headers = list(data.columns.values)

print(data.head())
print("\n")
print(data.tail())

print("\n")
print(data.columns)

print(data["review"][2])
print(data.at[2, "review"])

print(data.describe())

categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print(categorical_columns)
print(numerical_columns)

print(data[categorical_columns].describe())


for c in categorical_columns:
    print (data[c].unique())

print(str(data.count(axis=0)))

data_describe = data.describe(include=[object])
binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print("Binary: " + str(binary_columns))
print("Non-binary: " + str(nonbinary_columns))


for c in binary_columns:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[numpy.logical_not(top_items), c] = 1

print(data[binary_columns].describe())
print(data.head())

vectorizer = CountVectorizer(stop_words="english")
# vectorizer = CountVectorizer(stop_words="english")

X = data["review"]
y = data["sentiment"]
X = vectorizer.fit_transform(X)
X = X.astype(int)
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)


logreg = LogisticRegression(random_state=0, max_iter=10000)
logreg.fit(X_train, y_train)
print("P : " + str(precision_score(logreg.predict(X_test), y_test)))
print("R : " + str(recall_score(logreg.predict(X_test), y_test)))
print("F1: " + str(f1_score(logreg.predict(X_test), y_test)))
print("A : " + str(accuracy_score(logreg.predict(X_test), y_test)))