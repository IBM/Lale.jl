using Lale
import Pandas
using DataFrames
using PyCall

fetch = pyimport("lale.datasets.openml").fetch
(train_X, train_y), (test_X, test_y) = fetch("breast-cancer", "classification", preprocess=true)
train_X = DataFrame(train_X,:auto)
test_X = DataFrame(test_X,:auto)

MinMaxScaler = laleoperator("MinMaxScaler")
PCA = laleoperator("PCA")
NoOp = laleoperator("NoOp","lale")
LR= laleoperator("LogisticRegression")
RF= laleoperator("RandomForestClassifier")
SMOTE = laleoperator("SMOTE","imblearn")
CondensedNearestNeighbour = laleoperator("CondensedNearestNeighbour","imblearn")
SMOTEENN = laleoperator("SMOTEENN","imblearn")
accuracy_score=pyimport("sklearn.metrics").accuracy_score
Hyperopt = laleoperator("Hyperopt","lale")

pipeline_without_correction =  MinMaxScaler() >> PCA() >> RF()
optimizer = Hyperopt(estimator=pipeline_without_correction.model[:laleobj], max_evals = 10, scoring="roc_auc")
trained_optimizer = fit(optimizer,train_X, train_y)
predictions = predict(trained_optimizer,test_X)
trained_optimizer.model[:laleobj].summary()
score(:accuracy,test_y,predictions)

# SMOTE
pipeline_with_correction =  SMOTE(MinMaxScaler() >> PCA() >> RF())
optimizer = Hyperopt(estimator=pipeline_with_correction.model[:laleobj], max_evals = 10, scoring="roc_auc")
trained_optimizer = fit(optimizer,train_X, train_y)
predictions = predict(trained_optimizer,test_X)
trained_optimizer.model[:laleobj].summary()
score(:accuracy,test_y,predictions)

# CondensedNearestNeighbour
pipeline_with_correction =  CondensedNearestNeighbour(MinMaxScaler() >> PCA() >> RF())
optimizer = Hyperopt(estimator=pipeline_with_correction.model[:laleobj], max_evals = 10, scoring="roc_auc")
trained_optimizer = fit(optimizer,train_X, train_y)
predictions = predict(trained_optimizer,test_X)
trained_optimizer.model[:laleobj].summary()
score(:accuracy,test_y,predictions)

# SMOTEENN
pipeline_with_correction =  SMOTEENN(MinMaxScaler() >> PCA() >> RF())
optimizer = Hyperopt(estimator=pipeline_with_correction.model[:laleobj], max_evals = 10, scoring="roc_auc")
trained_optimizer = fit(optimizer,train_X, train_y)
predictions = predict(trained_optimizer,test_X)
trained_optimizer.model[:laleobj].summary()
score(:accuracy,test_y,predictions)
