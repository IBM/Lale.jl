module LalePreprocessors

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
export LalePreprocessor, lalepreprocessors

const preprocessor_dict = Dict{String,PyObject}()
const PREP = PyNULL()
const AGEN = PyNULL()
const LLIBS =PyNULL()


function __init__()
   copy!(PREP, pyimport("lale.lib.sklearn"))
   copy!(AGEN, pyimport("lale.lib.autogen"))
   copy!(LLIBS, pyimport("lale.lib.lale"))
   # Available lale scikit-learn learners.
   preprocessor_dict["NoOp"]                        = LLIBS.NoOp 
   preprocessor_dict["FeatureAgglomeration"]        = PREP.FeatureAgglomeration 
   preprocessor_dict["FactorAnalysis"]              = AGEN.FactorAnalysis
   preprocessor_dict["FastICA"]                     = AGEN.FastICA
   preprocessor_dict["IncrementalPCA"]              = AGEN.IncrementalPCA
   preprocessor_dict["KernelPCA"]                   = AGEN.KernelPCA
   preprocessor_dict["LatentDirichletAllocation"]   = AGEN.LatentDirichletAllocation
   preprocessor_dict["MiniBatchDictionaryLearning"] = AGEN.MiniBatchDictionaryLearning
   preprocessor_dict["MiniBatchSparsePCA"]          = AGEN.MiniBatchSparsePCA
   preprocessor_dict["NMF"]                         = PREP.NMF
   preprocessor_dict["PCA"]                         = PREP.PCA
   preprocessor_dict["SparsePCA"]                   = AGEN.SparsePCA
   preprocessor_dict["TruncatedSVD"]                = AGEN.TruncatedSVD
   preprocessor_dict["dictionary_learning"]         = AGEN.dictionary_learning
   preprocessor_dict["fast_ica"]                    = AGEN.fast_ica
   preprocessor_dict["SelectKBest"]                 = PREP.SelectKBest
   preprocessor_dict["RFE"]                         = PREP.RFE
   preprocessor_dict["SimpleImputer"]               = PREP.SimpleImputer
   preprocessor_dict["MissingIndicator"]            = PREP.MissingIndicator
   preprocessor_dict["Binarizer"]                   = AGEN.Binarizer
   preprocessor_dict["FunctionTransformer"]         = AGEN.FunctionTransformer
   preprocessor_dict["KBinsDiscretizer"]            = AGEN.KBinsDiscretizer
   preprocessor_dict["LabelBinarizer"]              = AGEN.LabelBinarizer
   preprocessor_dict["LabelEncoder"]                = AGEN.LabelEncoder
   preprocessor_dict["MultiLabelBinarizer"]         = AGEN.MultiLabelBinarizer
   preprocessor_dict["MaxAbsScaler"]                = AGEN.MaxAbsScaler
   preprocessor_dict["MinMaxScaler"]                = AGEN.MinMaxScaler
   preprocessor_dict["Normalizer"]                  = PREP.Normalizer
   preprocessor_dict["OneHotEncoder"]               = PREP.OneHotEncoder
   preprocessor_dict["OrdinalEncoder"]              = PREP.OrdinalEncoder
   preprocessor_dict["PolynomialFeatures"]          = PREP.PolynomialFeatures
   preprocessor_dict["PowerTransformer"]            = AGEN.PowerTransformer
   preprocessor_dict["QuantileTransformer"]         = PREP.QuantileTransformer
   preprocessor_dict["RobustScaler"]                = PREP.RobustScaler
   preprocessor_dict["StandardScaler"]              = PREP.StandardScaler
   #"IterativeImputer" => IMP.IterativeImputer,
   #"KNNImputer" => IMP.KNNImputer,
   #"add_dummy_feature" => PREP.add_dummy_feature,
   #"binarize" => PREP.binarize,
   #"label_binarize" => PREP.label_binarize,
   #"maxabs_scale" => PREP.maxabs_scale,
   #"minmax_scale" => PREP.minmax_scale,
   #"normalize" => PREP.normalize,
   #"quantile_transform" => PREP.quantile_transform,
   #"robust_scale" => PREP.robust_scale,
   #"scale" => PREP.scale,
   #"power_transform" => PREP.power_transform
end

"""
    LalePreprocessor(preprocessor::String,args::Dict=Dict())

A wrapper for Scikitlearn preprocessor functions. 
Invoking `skpreprocessors()` will list the acceptable 
and supported functions. Please check Scikitlearn
documentation for arguments to pass.

Implements `fit!` and `transform!`.
"""
mutable struct LalePreprocessor <: LaleOperator
   name::String
   model::Dict{Symbol,Any}

   function LalePreprocessor(args=Dict())
      default_args=Dict(
         :name => "laleprep",
         :preprocessor => "PCA",
         :autocomponent=>false,
         :impl_args => Dict()
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      prep = cargs[:preprocessor]
      if !(prep in keys(preprocessor_dict)) 
         println("$prep is not supported.") 
         println()
         skpreprocessors()
         throw(ArgumentError("Argument keyword error"))
      end
      impl_args = cargs[:impl_args]
      preprocessor = cargs[:preprocessor]
      py_preprocessor = preprocessor_dict[preprocessor]
      cargs[:laleobj] = py_preprocessor
      new(cargs[:name],cargs)
   end
end

function LalePreprocessor(prep::String,args::Dict)
  LalePreprocessor(Dict(:preprocessor => prep,:name=>prep,args...))
end

function LalePreprocessor(prep::String; args...)
   LalePreprocessor(Dict(:preprocessor => prep,:name=>prep,:impl_args=>Dict(pairs(args))))
end

function skpreprocessors()
  processors = keys(preprocessor_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: LalePreprocessor(name::String, args::Dict=Dict())")
  println("where *name* can be one of:")
  println()
  [print(processor," ") for processor in processors]
  println()
  println()
  println("and *args* are the corresponding preprocessor's initial parameters.")
  println("Note: Please consult Scikitlearn's online help for more details about the preprocessor's arguments.")
end

function fit!(skp::LalePreprocessor, x::DataFrame, y::Vector=[])
   features = x |> Array
   impl_args = copy(skp.model[:impl_args])
   autocomp = skp.model[:autocomponent]
   if autocomp == true
      cols = ncol(x)
      ncomponents = 1
      if cols > 0
         ncomponents = round(sqrt(cols),digits=0) |> Integer
         push!(impl_args,:n_components => ncomponents)
      end
   end
   preprocessor = skp.model[:preprocessor]
   py_preprocessor = preprocessor_dict[preprocessor]

   # Train model
   preproc = py_preprocessor(;impl_args...)
   preproc.fit(features)
   skp.model[:laleobj] = preproc
   skp.model[:impl_args] = impl_args
end

function transform!(skp::LalePreprocessor, x::DataFrame)
   features = deepcopy(x) |> Array
   model=skp.model[:laleobj]
   return collect(model.transform(features)) |> x->DataFrame(x,:auto)
end

function fit(skp::LalePreprocessor, x::DataFrame, y::Vector=[])
   features = x |> Array
   impl_args = copy(skp.model[:impl_args])
   autocomp = skp.model[:autocomponent]
   if autocomp == true
      cols = ncol(x)
      ncomponents = 1
      if cols > 0
         ncomponents = round(sqrt(cols),digits=0) |> Integer
         push!(impl_args,:n_components => ncomponents)
      end
   end
   preprocessor = skp.model[:preprocessor]
   py_preprocessor = preprocessor_dict[preprocessor]

   # Train model
   preproc = py_preprocessor(;impl_args...)
   preproc.fit(features)
   skp.model[:laleobj] = preproc
   skp.model[:impl_args] = impl_args
end

function transform(skp::LalePreprocessor, x::DataFrame)
   features = deepcopy(x) |> Array
   model=skp.model[:laleobj]
   return collect(model.transform(features)) |> x->DataFrame(x,:auto)
end


end

