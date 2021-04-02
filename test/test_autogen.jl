module TestAutoGen

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

#"DecisionTreeClassifier", ,"GradientBoostingClassifier",,"SGDClassifier",

const classifiers=["AdaBoostClassifier", "ExtraTreesClassifier","KNeighborsClassifier","LinearSVC", "MLPClassifier","MultinomialNB","RandomForestClassifier","RidgeClassifier","SVC"]

function fit_test(learner::String,in::DataFrame,out::Vector)
   _learner=LaleOp(learner,"autogen")
   fit!(_learner,in,out)
   @test _learner.model != Dict()
   return _learner
end

@testset "lale autogen classifiers" begin
   Random.seed!(123)
   for cl in classifiers
      println(cl)
      fit_test(cl,XC,YC)
   end
end

#,"SGDRegressor""GradientBoostingRegressor","DecisionTreeRegressor", 

const regressors = ["SVR", "RandomForestRegressor","KNeighborsRegressor",  "AdaBoostRegressor", "ExtraTreesRegressor"]
function fit_transform_reg(model::LaleOp,in::DataFrame,out::Vector)
   @test sum((transform(model,in) .- out).^2)/length(out) < 2.0
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
