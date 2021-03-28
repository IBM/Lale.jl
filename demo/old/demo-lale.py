# python import
import sklearn
import pandas as pd
import lale.datasets as dt
import lale
from lale.lib.sklearn import RandomForestRegressor as RF
from lale.lib.sklearn import LinearRegression as LinReg
from lale.lib.sklearn import PCA
from lale.lib.autogen import FastICA as ICA
from lale.lib.sklearn import DecisionTreeRegressor as Tree
from lale.lib.sklearn.pipeline import Pipeline
from lale.lib.lale import Hyperopt  # GridSearchCV, NoOp, ConcatFeatures
# from lale.operators import make_pipeline, make_choice, make_union


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
data = load_iris()
X, y = data.data, data.target
y=X[:, 3]
X=X[:, 0:3]
X_train, X_test, y_train, y_test = train_test_split(X, y)


# load data
(train_X, train_y), (test_X, test_y) = dt.california_housing_df()
pd.concat([train_X.head(), train_y.head()], axis=1)
lale.wrap_imported_operators()

# pipeline 1
pca_tree_planned = Pipeline(steps=[("tfm", PCA()), ("estim", Tree())])
pca_tree_planned.fit(train_X, train_y)
predicted = pca_tree_planned.predict(test_X)
print(f'R2 score {sklearn.metrics.r2_score(test_y, predicted):.2f}')

# pipeline 2
pca_tree_planned = PCA() >> Tree()
pca_tree_trained = pca_tree_planned.auto_configure(
    train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=10, verbose=True)
predicted = pca_tree_trained.predict(test_X)
print(f'R2 score {sklearn.metrics.r2_score(test_y, predicted):.2f}')

# iris data
pca_tree_planned = PCA >> RF
pca_tree_trained = pca_tree_planned.auto_configure(
    X, y, optimizer=Hyperopt, cv=3, max_evals=10, verbose=True)
predicted = pca_tree_trained.predict(X)
print(f'R2 score {sklearn.metrics.r2_score(y, predicted):.2f}')
mean_squared_error(y, predicted)

# pipeline 3
# planned    = make_pipeline(make_union(PCA() , NoOp()), make_choice(LinReg(),  Tree()))
planned    = (PCA() & NoOp())  >> ConcatFeatures >> (LinReg() | Tree())
trained    = planned.auto_configure(train_X, train_y, optimizer=Hyperopt, cv=3,  max_evals=3, verbose=True)
predicted  = trained.predict(test_X)
print(f'R2 score {sklearn.metrics.r2_score(test_y, predicted):.2f}')


# pipeline 4
planned    = (PCA(svd_solver="auto") & NoOp() ) >> ConcatFeatures() >> (LinReg() | Tree())
trained    = planned.auto_configure(train_X, train_y,
                                    optimizer=Hyperopt,
                                    cv=3,max_evals=10,verbose=True)
predicted  = trained.predict(test_X)
print(f'R2 score {sklearn.metrics.r2_score(test_y, predicted):.2f}')


# pipeline 5 
pipeline    = ((PCA(svd_solver="auto") & ICA()) >> ConcatFeatures()) >> Tree()
trainedgrid = pipeline.auto_configure(train_X, train_y, optimizer=Hyperopt,max_evals=5, cv=3,verbose=True)
predicted   = trainedgrid.predict(test_X);
print(f'R2 score {sklearn.metrics.r2_score(test_y, predicted):.2f}')
