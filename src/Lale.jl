module Lale

using PyCall
using Pandas
using Conda

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
   export LaleOperator
   abstract type LaleOperator <: Learner end
end

include("lalelearner.jl")
using .LaleLearners
export LaleLearner, lalelearners

include("lalepreprocessor.jl")
using .LalePreprocessors
export LalePreprocessor, lalepreprocessor

include("lalelibops.jl")
using .LaleLibOps
export NoOp
export LaleOptimizer, laleoptimizers
export >>, +, |, |>, &

end
