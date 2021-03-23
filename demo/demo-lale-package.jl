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
pca     = LalePreprocessor("PCA")
rb      = LalePreprocessor("RobustScaler")
noop    = LalePreprocessor("NoOp")
rfr     = LaleLearner("RandomForestRegressor")
rfc     = LaleLearner("RandomForestClassifier")
treereg = LaleLearner("DecisionTreeRegressor")

pca |> typeof |> supertypes

rfr |> typeof |> supertypes

# amlp ops
ohe  = OneHotEncoder()
catf = CatFeatureSelector()
numf = NumFeatureSelector()

numf |> typeof |> supertypes

# regression
lalepipe =  (pca + noop) >>  (rfr | treereg )
lale_hopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals=10,cv=3)
lalepred = fit_transform!(lale_hopt,Xreg,Yreg)
lalermse=score(:rmse,lalepred,Yreg)

lalepipe |> typeof
lale_hopt |> typeof |> supertypes

amlpipe = @pipeline  (pca + noop) |> (rfr * treereg)
amlpred = fit_transform!(amlpipe,Xreg,Yreg)
crossvalidate(amlpipe,Xreg,Yreg,"mean_squared_error")
amlprmse=score(:rmse,amlpred,Yreg)

amlpipe |> typeof |> supertypes


@test amlprmse < lalermse

# classification 
lalepipe =  (rb + pca) |> rfc
lale_hopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals = 10,cv = 3)
lalepred  = fit_transform!(lale_hopt,Xcl,Ycl)
laleacc   = score(:accuracy,lalepred,Ycl)

amlpipe = @pipeline  (pca + rb) |> rfc
amlpred = fit_transform!(amlpipe,Xcl,Ycl)
crossvalidate(amlpipe,Xcl,Ycl,"accuracy_score")
amlpacc = score(:accuracy,amlpred,Ycl)

@test abs(amlpacc - laleacc) < 5.0

plr = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfr;
crossvalidate(plr,Xreg,Yreg,"mean_absolute_error",10,false) 

plc = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfc;
crossvalidate(plc,Xcl,Ycl,"accuracy_score",10,false) 
