from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate
import pandas
import matplotlib
from sklearn.model_selection import train_test_split

input_file = "../../../resources/IMDB Dataset.csv"
data = pandas.read_csv(input_file, header = 0)
original_headers = list(data.columns.values)

print(data.head())
print(data.tail())

print("\n")
print(data.describe())

categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print(f"\nCategorical columns: {categorical_columns}")
print(f"Numerical columns: {numerical_columns}")

print("\n")
for c in categorical_columns:
    print (f"Unique '{c}' values: {len(data[c].unique())}")

#check for empty values
print(data.count(axis=0))

data_describe = data.describe(include=[object])
binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(f"\nBinary: {str(binary_columns)}")
print(f"Non-binary: {str(nonbinary_columns)}")


data.sentiment = data.sentiment.apply(lambda x: 0 if x == "negative" else 1)

print("\n")
print(data.head())

vectorizer = CountVectorizer()
# vectorizer = CountVectorizer(stop_words="english"б, ngram_range=(1,2), min_df=0.1, max_features = 100000)

X = data["review"]
y = data["sentiment"]
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11, stratify=y)
X_train, X_validate, y_train, y_validate = train_test_split(X_cv, y_cv, test_size = 0.25, random_state = 11, stratify=y_cv)

X_train = vectorizer.fit_transform(X_train)
X_validate = vectorizer.transform(X_validate)
X_test = vectorizer.transform(X_test)


logreg = LogisticRegression(random_state=0, max_iter=10000)
logreg.fit(X_train, y_train)

scores_validation = cross_validate(logreg, X_validate, y_validate, cv = 5, scoring=("f1", "accuracy", "precision", "recall", "roc_auc"))
print("\n")

print("P: %0.2f (+/- %0.3f)" % (scores_validation["test_precision"].mean(), scores_validation["test_precision"].std() / 2))
print("R: %0.2f (+/- %0.3f)" % (scores_validation["test_recall"].mean(), scores_validation["test_recall"].std() / 2))
print("F1: %0.2f (+/- %0.3f)" % (scores_validation["test_f1"].mean(), scores_validation["test_f1"].std() / 2))
print("A: %0.2f (+/- %0.3f)" % (scores_validation["test_accuracy"].mean(), scores_validation["test_accuracy"].std() / 2))
print("ROC_AUC: %0.2f (+/- %0.3f)" % (scores_validation["test_roc_auc"].mean(), scores_validation["test_roc_auc"].std() / 2))


scores_test = cross_validate(logreg, X_test, y_test, cv = 5, scoring=("f1", "accuracy", "precision", "recall", "roc_auc"))
print("\n")

prediction = logreg.predict(X_test)
print("P : %0.3f" % precision_score(prediction, y_test))
print("R : %0.3f" % recall_score(prediction, y_test))
print("F1: %0.3f" % f1_score(prediction, y_test))
print("A : %0.3f" % accuracy_score(prediction, y_test))
print("ROC_AUC %0.3f: " % roc_auc_score(prediction, y_test))


# analytics = {}
# vect_validation["CountVectorizer_val_roc_auc"]=scores_validation["test_roc_auc"].mean()
# vect_validation["CountVectorizer_val_f1"]=scores_validation["test_f1"].mean()
# vect_test["CountVectorizer_test_f1"]=f1_score(prediction, y_test)
# vect_test["CountVectorizer_test_roc_auc"]=roc_auc_score(prediction, y_test)
# analytics['CountVectorizer_val'] = vect_validation
# analytics['CountVectorizer_test'] = vect_test

analytics = {}
analytics['CountVectorizer_val'] = scores_validation["test_f1"].mean()
analytics['CountVectorizer_test'] = f1_score(prediction, y_test)

DataFrame.from_dict(data = analytics, orient='index').plot(kind='bar', legend=False)