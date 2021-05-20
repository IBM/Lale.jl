import sklearn
import pandas as pd
import lale.datasets as dt
import lale

from lale.lib.lale import Project
from lale.lib.sklearn import OneHotEncoder
from lale.lib.lale import ConcatFeatures
from lale.lib.sklearn import DecisionTreeClassifier as Tree
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.sklearn import RandomForestClassifier as RF
from lale.lib.sklearn import PCA
from lale.lib.lale import Hyperopt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.metrics
accuracy_scorer = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)

df = pd.read_csv("credit.csv")
y = df["class"]
X = df.iloc[:,1:]
train_X, test_X, train_y, test_y = train_test_split(X, y)

prep_to_numbers = (
    (Project(columns={"type": "string"}) >> OneHotEncoder(handle_unknown = "ignore"))
    & Project(columns={"type": "number"})
    ) >> ConcatFeatures
planned_orig = prep_to_numbers >> ( Tree | KNN)

best_estimator = planned_orig.auto_configure(
    train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=10)
print(f'accuracy {accuracy_scorer(best_estimator, test_X, test_y):.1%}')

pca_tree_planned = prep_to_numbers >> PCA >> RF
pca_tree_trained = pca_tree_planned.auto_configure(
    train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=10, verbose=True)
print(f'accuracy {accuracy_scorer(pca_tree_trained, test_X, test_y):.1%}')
