module Lale

using PyCall

# load abstract super-types and utils
using AMLPipelineBase
using AMLPipelineBase.AbsTypes
export fit!, transform!,fit_transform!

using AMLPipelineBase
using AMLPipelineBase: AbsTypes, Utils, BaselineModels, Pipelines
using AMLPipelineBase: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AMLPipelineBase: EnsembleMethods, CrossValidators
using AMLPipelineBase: NARemovers

export Machine, Learner, Transformer, Workflow, Computer
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples,
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris, getprofb,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing
export Baseline, Identity
export Imputer,OneHotEncoder,Wrapper
export PrunedTree,RandomForest,Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline, @pipelinex
export Pipeline, ComboPipeline

import AMLPipelineBase.AbsTypes: fit!, transform!

# ------
module LaleAbsTypes
   using ..AbsTypes
   using DataFrames

   export LaleOperator, fit, transform

   abstract type LaleOperator <: Learner end
   fit(o::LaleOperator, x::DataFrame, y::Vector) = nothing
   tranform(o::LaleOperator, x::DataFrame) = nothing
end

include("laleop.jl")
using .LaleOps
export LaleOp, skops, autogenops, lalelibops

include("lalelibop.jl")
using .LaleLibOps
export LaleOptimizer, laleoptimizers
export >>, +, |, |>, &

export laleoperator
export fit, transform, fit!, transform!

function laleoperator(name::String,type::String="sklearn"; args...)
   try
      obj = LaleOp(name,type;args...)
      return obj
   catch ArgumentError
      sk = keys(LaleOps.sk_dict)
      ag = keys(LaleOps.ag_dict)
      ll = keys(LaleOps.ll_dict)
      println("Please choose among these pipeline elements:")
      println()
      println("sklearn: ", [sk...])
      println()
      println("autogen: ", [ag...])
      println()
      println("lale: ",    [ll...])
      println()
      throw(ArgumentError("$name does not exist"))
   end
end

end
