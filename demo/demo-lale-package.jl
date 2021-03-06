# LaleOperator abstract type is a subtype 
# of the AutoMLPipeline Machine supertype
# 
# abstract type Machine end
# abstract type Computer     <: Machine  end
# abstract type Workflow     <: Machine  end
# abstract type Learner      <: Computer end
# abstract type Transformer  <: Computer end
#
# abstract type LaleOperator <: Learner end
#
# typeof(LaleLearner)        <: LaleOperator
# typeof(LalePreprocessor)   <: LaleOperator
# typeof(LaleOptimizer)      <: LaleOperator
# ----------------------------------------

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
# amlp ops
ohe  = OneHotEncoder()
catf = CatFeatureSelector()
numf = NumFeatureSelector()

# Lale regression
lalepipe    = (pca + noop) >>  (rfr | treereg )
lale_hopt   = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
laletrained = fit(lale_hopt,Xreg,Yreg)
lalepred    = transform(laletrained,Xreg)
lalermse    = score(:rmse,lalepred,Yreg)

# AutoMLPipeline regression
amlpipe = @pipeline  (pca + noop) |> (rfr * treereg)
amlpred = fit_transform!(amlpipe,Xreg,Yreg)
crossvalidate(amlpipe,Xreg,Yreg,"mean_squared_error")
amlprmse = score(:rmse,amlpred,Yreg)

# Lale classification 
lalepipe    = (rb + pca) |> rfc
lale_hopt   = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
laletrained = fit(lale_hopt,Xcl,Ycl)
lalepred    = transform(laletrained,Xcl)
laleacc     = score(:accuracy,lalepred,Ycl)

# AutoMLPipeline classification
amlpipe = @pipeline  (pca + rb) |> rfc
amlpred = fit_transform!(amlpipe,Xcl,Ycl)
crossvalidate(amlpipe,Xcl,Ycl,"accuracy_score")
amlpacc = score(:accuracy,amlpred,Ycl)

# AutoMLPipeline regression
plr = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfr;
crossvalidate(plr,Xreg,Yreg,"mean_absolute_error",10,false) 

#AutoMLPipeline classification
plc = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfc;
crossvalidate(plc,Xcl,Ycl,"accuracy_score",10,false) 
