using Lale
using InteractiveUtils

using Random
using Statistics
using Test
using DataFrames: DataFrame
using AutoMLPipeline: Utils

iris = getiris()
Xreg = iris[:,1:3] |> DataFrame
Yreg = iris[:,4]   |> Vector
Xcl  = iris[:,1:4] |> DataFrame
Ycl  = iris[:,5]   |> Vector

# lale ops
pca     = laleoperator("PCA")
rb      = laleoperator("RobustScaler")
noop    = laleoperator("NoOp")
rfr     = laleoperator("RandomForestRegressor")
rfc     = laleoperator("RandomForestClassifier")
treereg = laleoperator("DecisionTreeRegressor")
# amlp ops
ohe  = OneHotEncoder()
catf = CatFeatureSelector()
numf = NumFeatureSelector()

# Lale regression
lalepipe =  (pca + noop) >>  (rfr | treereg )
lale_hopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals=10,cv=3)
fit(lale_hopt,Xreg,Yreg)
lalepred = transform(lale_hopt,Xreg)
lalermse=score(:rmse,lalepred,Yreg)

# Lale classification 
lalepipe =  (rb + pca) |> rfc
lale_hopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals = 10,cv = 3)
fit(lale_hopt,Xcl,Ycl)
lalepred = transform(lale_hopt,Xcl)
laleacc   = score(:accuracy,lalepred,Ycl)
