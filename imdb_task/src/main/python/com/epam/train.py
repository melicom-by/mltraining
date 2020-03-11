from collections import Counter
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate
import pandas
import matplotlib
from sklearn.model_selection import train_test_split

most_common_duplicates_count = 5
random_state_value = 11

# Data import
input_file = "../../../resources/IMDB Dataset.csv"
data = pandas.read_csv(input_file, header = 0)
original_headers = list(data.columns.values)


# Exploratory data analysis
print(data.head())

## Looking for unique values
print("\n")
for c in original_headers:
    print (f"Unique '{c}' values: {len(data[c].unique())}")

## Analyzing the most common duplicates and removing them
duplicates_counter = Counter(data[data.duplicated()]["review"])
most_common_duplicates = duplicates_counter.most_common(most_common_duplicates_count)
for review, count in most_common_duplicates:
    print('%d : %s \n' % (count + 1, review))
data = data.drop_duplicates(keep='first')


## Check for empty values
print("\n")
print(data.count(axis=0))

# Vectorization
## Replace categories with digits
data.sentiment = data.sentiment.apply(lambda x: 0 if x == "negative" else 1)

print("\n")
print(data.head())

vectorizer = CountVectorizer()
# vectorizer = CountVectorizer(stop_words="english"б, ngram_range=(1,2), min_df=0.1, max_features = 100000)

## Data split for Train-validate-test pipeline. Proportion is 80/20/20
X = data["review"]
y = data["sentiment"]
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size = 0.2, random_state=random_state_value, stratify=y)
X_train, X_validate, y_train, y_validate = train_test_split(X_cv, y_cv, test_size = 0.25, random_state = random_state_value, stratify=y_cv)

## Vocabulary creation
X_train = vectorizer.fit_transform(X_train)
X_validate = vectorizer.transform(X_validate)
X_test = vectorizer.transform(X_test)

# Model training
logreg = LogisticRegression(random_state=random_state_value, max_iter=10000)
logreg.fit(X_train, y_train)

## Cross-validation on validation dataset
scores_validation = cross_validate(logreg, X_validate, y_validate, cv = 5, scoring=("f1", "accuracy", "precision", "recall", "roc_auc"))
print("\n")

def print_cv_score(source, scoreName, abbr):
    print(abbr + ": %0.2f (+/- %0.3f)" % (source[scoreName].mean(), source[scoreName].std() / 2))

print_cv_score(scores_validation, "test_precision", "P")
print_cv_score(scores_validation, "test_recall", "R")
print_cv_score(scores_validation, "test_f1", "F1")
print_cv_score(scores_validation, "test_accuracy", "A")
print_cv_score(scores_validation, "test_roc_auc", "ROC_AUC")


# Testing on the independent test set
prediction = logreg.predict(X_test)
print("\n")

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


#Analytics report creation. Comparing algorithms and experiments.
analytics = {}
analytics['CountVectorizer_val'] = scores_validation["test_f1"].mean()
analytics['CountVectorizer_test'] = f1_score(prediction, y_test)

DataFrame.from_dict(data = analytics, orient='index').plot(kind='bar', legend=False)