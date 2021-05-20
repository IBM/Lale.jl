from lale.lib.aif360 import fetch_creditg_df
all_X, all_y, fairness_info = fetch_creditg_df()

import pandas as pd
pd.options.display.max_columns = None
pd.concat([all_y, all_X], axis=1)

import lale.pretty_print
lale.pretty_print.ipython_display(fairness_info)

from lale.lib.aif360 import fair_stratified_train_test_split
train_X, test_X, train_y, test_y = fair_stratified_train_test_split(
    all_X, all_y, **fairness_info, test_size=0.33, random_state=42)

from lale.lib.aif360 import disparate_impact
disparate_impact_scorer = disparate_impact(**fairness_info)
print("disparate impact of training data {:.2f}, test data {:.2f}".format(
    disparate_impact_scorer.scoring(X=train_X, y_pred=train_y),
    disparate_impact_scorer.scoring(X=test_X, y_pred=test_y)))

from lale.lib.lale import Project
from sklearn.preprocessing import OneHotEncoder
from lale.lib.lale import ConcatFeatures
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.neighbors import KNeighborsClassifier as KNN

import lale
lale.wrap_imported_operators()
prep_to_numbers = (
    (Project(columns={"type": "string"}) >> OneHotEncoder(handle_unknown="ignore"))
    & Project(columns={"type": "number"})
    ) >> ConcatFeatures
planned_orig = prep_to_numbers >> (LR | Tree | KNN)
planned_orig.visualize()

from lale.lib.lale import Hyperopt
best_estimator = planned_orig.auto_configure(
    train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=10)
best_estimator.visualize()

best_estimator.pretty_print(ipython_display=True, show_imports=False)

import sklearn.metrics
accuracy_scorer = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
print(f'accuracy {accuracy_scorer(best_estimator, test_X, test_y):.1%}')

print(f'disparate impact {disparate_impact_scorer(best_estimator, test_X, test_y):.2f}')

from lale.lib.aif360 import DisparateImpactRemover
lale.pretty_print.ipython_display(
    DisparateImpactRemover.hyperparam_schema('repair_level'))

di_remover = DisparateImpactRemover(
    **fairness_info, preparation=prep_to_numbers)
planned_fairer = di_remover >> (LR | Tree | KNN)
planned_fairer.visualize()

from lale.lib.aif360 import accuracy_and_disparate_impact
combined_scorer = accuracy_and_disparate_impact(**fairness_info)

from lale.lib.aif360 import FairStratifiedKFold
fair_cv = FairStratifiedKFold(**fairness_info, n_splits=3)

trained_fairer = planned_fairer.auto_configure(
    train_X, train_y, optimizer=Hyperopt, cv=fair_cv,
    max_evals=10, scoring=combined_scorer, best_score=1.0)

print(f'accuracy {accuracy_scorer(trained_fairer, test_X, test_y):.1%}')
print(f'disparate impact {disparate_impact_scorer(trained_fairer, test_X, test_y):.2f}')
trained_fairer.visualize()

trained_fairer.pretty_print(ipython_display=True, show_imports=False)

