module LaleLearners

using PyCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..LaleAbsTypes
using ..Utils

import ..AbsTypes: fit!, transform!
import ..LaleAbsTypes: fit, transform

export fit!, transform!, fit, transform
export LaleLearner, lalelearners

const learner_dict = Dict{String,PyObject}() 
const LSKL   = PyNULL()


function __init__()
   copy!(LSKL, pyimport("lale.lib.sklearn"))

   # Available scikit-learn learners.
   learner_dict["AdaBoostClassifier"]          = LSKL.AdaBoostClassifier
   learner_dict["BaggingClassifier"]           = LSKL.BaggingClassifier
   learner_dict["DecisionTreeClassifier"]      = LSKL.DecisionTreeClassifier
   learner_dict["ExtraTreesClassifier"]        = LSKL.ExtraTreesClassifier
   learner_dict["GradientBoostingClassifier"]  = LSKL.GradientBoostingClassifier
   learner_dict["RandomForestClassifier"]      = LSKL.RandomForestClassifier
   learner_dict["KNeighborsClassifier"]        = LSKL.KNeighborsClassifier
   learner_dict["LinearSVC"]                   = LSKL.LinearSVC
   learner_dict["MLPClassifier"]               = LSKL.MLPClassifier
   learner_dict["PassiveAggressiveClassifier"] = LSKL.PassiveAggressiveClassifier
   learner_dict["RidgeClassifier"]             = LSKL.RidgeClassifier
   learner_dict["SGDClassifier"]               = LSKL.SGDClassifier
   learner_dict["SVC"]                         = LSKL.SVC
   learner_dict["VotingClassifier"]            = LSKL.VotingClassifier
   learner_dict["QDA"]                         = LSKL.QuadraticDiscriminantAnalysis
   learner_dict["LogisticRegression"]          = LSKL.LogisticRegression
   learner_dict["GaussianNB"]                  = LSKL.GaussianNB
   learner_dict["MultinomialNB"]               = LSKL.MultinomialNB
   learner_dict["SVR"]                         = LSKL.SVR
   learner_dict["Ridge"]                       = LSKL.Ridge
   learner_dict["SGDRegressor"]                = LSKL.SGDRegressor
   learner_dict["KNeighborsRegressor"]         = LSKL.KNeighborsRegressor
   learner_dict["DecisionTreeRegressor"]       = LSKL.DecisionTreeRegressor
   learner_dict["RandomForestRegressor"]       = LSKL.RandomForestRegressor
   learner_dict["ExtraTreesRegressor"]         = LSKL.ExtraTreesRegressor
   learner_dict["AdaBoostRegressor"]           = LSKL.AdaBoostRegressor
   learner_dict["GradientBoostingRegressor"]   = LSKL.GradientBoostingRegressor
end

"""
    LaleLearner(learner::String, args::Dict=Dict())

A Scikitlearn wrapper to load the different machine learning models.
Invoking `lalelearners()` will list the available learners. Please
consult Scikitlearn documentation for arguments to pass.

Implements `fit!` and `transform!`. 
"""
mutable struct LaleLearner <: LaleOperator
   name::String
   model::Dict{Symbol,Any}

   function LaleLearner(args=Dict{Symbol,Any}())
      default_args=Dict{Symbol,Any}(
         :name => "lalelearner",
         :output => :class,
         :learner => "LinearSVC",
         :impl_args => Dict{Symbol,Any}()
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      lr = cargs[:learner]
      if !(lr in keys(learner_dict)) 
         println("$lr is not supported.") 
         println()
         lalelearners()
         throw(ArgumentError("Argument keyword error"))
      end
      impl_args = cargs[:impl_args]
      learner = cargs[:learner]
      py_learner = learner_dict[learner]
      cargs[:laleobj]   = py_learner
      new(cargs[:name],cargs)
   end
end

function LaleLearner(learner::String, args::Dict)
   LaleLearner(Dict(:learner => learner,:name=>learner, args...))
end

function LaleLearner(learner::String; args...)
   LaleLearner(Dict(:learner => learner,:name=>learner,:impl_args=>Dict(pairs(args))))
end

"""
    function lalelearners()

List the available scikitlearn machine learners.
"""
function lalelearners()
  learners = keys(learner_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: LaleLearner(name::String, args::Dict=Dict())")
  println("where 'name can be one of:")
  println()
  [print(learner," ") for learner in learners]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Scikitlearn's online help for more details about the learner's arguments.")
end

function fit!(lale::LaleLearner, xx::DataFrame, y::Vector)
  x = xx |> Array
  impl_args = copy(lale.model[:impl_args])
  learner = lale.model[:learner]
  py_learner = learner_dict[learner]

  # Assign CombineML-specific defaults if required
  if learner == "RadiusNeighborsClassifier"
    if get(impl_args, :outlier_label, nothing) == nothing
      impl_options[:outlier_label] = labels[rand(1:size(labels, 1))]
    end
  end

  # Train
  modelobj = py_learner(;impl_args...)
  modelobj.fit(x,y)
  lale.model[:laleobj]   = modelobj
  lale.model[:impl_args] = impl_args
end


function transform!(lale::LaleLearner, xx::DataFrame)
   x = deepcopy(xx)|> Array
   #return collect(learner.model[:predict](x))
   lalelearner = lale.model[:laleobj]
   return collect(lalelearner.predict(x))
end

fit(lale::LaleLearner, xx::DataFrame, y::Vector) = fit!(lale,xx,y)

transform(lale::LaleLearner, xx::DataFrame)=transform!(lale,xx)


end

