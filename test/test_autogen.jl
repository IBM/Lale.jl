module TestLaleOp

using Random
using Test
using Lale
using Statistics
using DataFrames: DataFrame

const IRIS = getiris()
const X = IRIS[:,1:3] |> DataFrame
const XC = IRIS[:,1:4] |> DataFrame
const YC = IRIS[:,5] |> Vector
const Y = IRIS[:,4] |> Vector

const classifiers=["AdaBoostClassifier", "DecisionTreeClassifier", "ExtraTreesClassifier","GradientBoostingClassifier","KNeighborsClassifier","LinearSVC", "MLPClassifier","MultinomialNB","RandomForestClassifier","RidgeClassifier","SGDClassifier","SVC"]

function fit_test(learner::String,in::DataFrame,out::Vector)
   _learner=LaleOp(learner,"autogen")
   fit!(_learner,in,out)
   @test _learner.model != Dict()
   return _learner
end

@testset "lale autogen classifiers" begin
   Random.seed!(123)
   for cl in classifiers
      fit_test(cl,XC,YC)
   end
end

const regressors = ["SVR", "RandomForestRegressor","SGDRegressor","KNeighborsRegressor", "GradientBoostingRegressor", "AdaBoostRegressor", "DecisionTreeRegressor", "ExtraTreesRegressor"]

function fit_transform_reg(model::LaleOp,in::DataFrame,out::Vector)
   @test sum((transform!(model,in) .- out).^2)/length(out) < 2.0
end
@testset "lale autogen regressors" begin
   Random.seed!(123)
   for rg in regressors
      println(rg)
      model=fit_test(rg,X,Y)
      fit_transform_reg(model,X,Y)
   end
end


end
