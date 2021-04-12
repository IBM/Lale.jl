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
rb      = laleoperator("RobustScaler","autogen")
noop    = laleoperator("NoOp","lale")
rfr     = laleoperator("RandomForestRegressor")
rfc     = laleoperator("RandomForestClassifier")
treereg = laleoperator("DecisionTreeRegressor")

# Lale regression
lalepipe =  (pca + noop) >>  (rfr | treereg )
lale_hopt = LalePipeOptimizer(lalepipe,max_evals=50,cv=3)
laletrained=fit(lale_hopt,Xreg,Yreg)
lalepred = transform(laletrained,Xreg)
lalermse=score(:rmse,lalepred,Yreg)

# AutoMLPipeline regression
amlpipe = @pipeline  (pca + noop) |> (rfr * treereg)
fit!(amlpipe,Xreg,Yreg)
amlpred = transform!(amlpipe,Xreg)
amlprmse=score(:rmse,amlpred,Yreg)


# Lale classification 
lalepipe =  (rb + pca) |> rfc
lale_hopt = LalePipeOptimizer(lalepipe, max_evals = 10,cv = 3)
laletrained=fit(lale_hopt,Xcl,Ycl)
lalepred = transform(laletrained,Xcl)
laleacc   = score(:accuracy,lalepred,Ycl)

# AutoMLPipeline classification
amlpipe = @pipeline  (pca + rb) |> rfc
amlpred = fit!(amlpipe,Xcl,Ycl)
amlpred = transform!(amlpipe,Xcl)
amlpacc = score(:accuracy,amlpred,Ycl)
