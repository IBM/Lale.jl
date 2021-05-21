module LaleOps

using PyCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..LaleAbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!, lalepropertynames
import ..LaleAbsTypes: predict, pretty_print
import ..LaleLibOps: LalePipe

export fit!, transform!, fit, transform, predict
export LaleOp, skops, autogenops, lalelibops, pretty_print, lalesummary

const sk_dict = Dict{String,PyObject}() 
const ag_dict = Dict{String,PyObject}() 
const ll_dict = Dict{String,PyObject}() 
const mb_dict = Dict{String,PyObject}() 
const SK   = PyNULL()
const AG   = PyNULL()
const LL   = PyNULL()
const MB   = PyNULL()

include("sklearn-const.jl") # SKVec
include("autogen-const.jl") # AGVec
include("liblale-const.jl") # LLVec
include("imblearn-const.jl") # MBVec


function __init__()
   copy!(SK, pyimport("lale.lib.sklearn"))
   copy!(AG, pyimport("lale.lib.autogen"))
   copy!(LL, pyimport("lale.lib.lale"))
   copy!(MB, pyimport("lale.lib.imblearn"))

   for lr in SKVec
      sk_dict[lr] = SK
   end
   for lr in AGVec
      ag_dict[lr] = AG
   end
   for lr in LLVec
      ll_dict[lr] = LL
   end
   for lr in MBVec
      mb_dict[lr] = MB
   end
end

"""
    LaleOp(learner::String, args::Dict=Dict())

A Scikitlearn wrapper to load the different machine learning models.
Invoking `skops()` or `autogenops()` will list the available learners. Please
consult Scikitlearn documentation for arguments to pass.

Implements `fit!` and `transform!`. 
"""
mutable struct LaleOp <: LaleOperator
   name::String
   model::Dict{Symbol,Any}

   function LaleOp(args=Dict{Symbol,Any}())
      default_args=Dict{Symbol,Any}(
         :name => "laleop",
         :type => "sklearn", # or "autogen" or "lale"
         :output => :class,
         :learner => "LinearSVC",
         :impl_args => Dict{Symbol,Any}()
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      lr = cargs[:learner]
      type = cargs[:type]
      if type == "sklearn"
         if !(lr in keys(sk_dict)) 
            println("$lr is not supported.") 
            println()
            skops()
            throw(ArgumentError("Argument keyword error"))
         end
      elseif type == "autogen"
         if !(lr in keys(ag_dict)) 
            println("$lr is not supported.") 
            println()
            autogenops()
            throw(ArgumentError("Argument keyword error"))
         end
      elseif type == "lale"
         if !(lr in keys(ll_dict)) 
            println("$lr is not supported.") 
            println()
            lalelibops()
            throw(ArgumentError("Argument keyword error"))
         end
      elseif type == "imblearn"
         if !(lr in keys(mb_dict)) 
            println("$lr is not supported.") 
            println()
            lalelibops()
            throw(ArgumentError("Argument keyword error"))
         end
      else
         throw(ArgumentError("Argument keyword error"))
      end

      impl_args = cargs[:impl_args]
      learner = cargs[:learner]
      if type == "sklearn"
         py_learner = getproperty(sk_dict[learner],learner)
      elseif type == "autogen"
         py_learner = getproperty(ag_dict[learner],learner)
      elseif type == "lale"
         py_learner = getproperty(ll_dict[learner],learner)
      elseif type == "imblearn"
         py_learner = getproperty(mb_dict[learner],learner)
      else
         throw(ArgumentError("$learner not found"))
      end
      cargs[:laleobj]   = py_learner
      new(cargs[:name],cargs)
   end
end

# initialized laleobject and pass the args for fit
function (x::LaleOp)(;args...) 
   pyobj = x.model[:laleobj]
   pr = pyobj(;args...)
   x.model[:laleobj] = pr
   x.model[:impl_args]=Dict(pairs(args))
   return x
end

# check if operator is Hyperopt or SMOTE,etc
function (x::LaleOp)(pipe::LalePipe; args...) 
   pyobj = x.model[:laleobj]
   lpipe = pipe.model[:laleobj]
   if x.model[:learner] == "Hyperopt" # use estimator arg
      pr = pyobj(estimator=lpipe;args...) 
      x.model[:laleobj] = pr
      x.model[:impl_args]=Dict(pairs(args))
      return x
   else #imbalance operator, use operator arg
      pr = pyobj(operator=lpipe;args...) |> LalePipe
      return pr
   end
end

function LaleOp(learner::String, args::Dict)
   LaleOp(Dict(:learner => learner,:name=>learner, args...))
end

function LaleOp(learner::String,type::String="sklearn"; args...)
   LaleOp(Dict(:learner => learner,:name=>learner,:type=>type,:impl_args=>Dict(pairs(args))))
end

"""
    function skops()

List the available scikitlearn machine learners.
"""
function skops()
  learners = keys(sk_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: LaleOp(name::String, type=\"sklearn\"; args...)")
  println("where 'name can be one of:")
  println()
  [print(learner," ") for learner in learners]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Scikitlearn's online help for more details about the learner's arguments.")
end

"""
    function autogenops()

List the available scikitlearn machine learners.
"""
function autogenops()
  learners = keys(ag_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: LaleOp(name::String, type=\"autogen\"; args...)")
  println("where 'name can be one of:")
  println()
  [print(learner," ") for learner in learners]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Scikitlearn's online help for more details about the learner's arguments.")
end

"""
    function lalelibops()

List the available scikitlearn machine learners.
"""
function lalelibops()
  learners = keys(ll_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: LaleOp(name::String, type=\"lale\"; args...)")
  println("where 'name can be one of:")
  println()
  [print(learner," ") for learner in learners]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Scikitlearn's online help for more details about the learner's arguments.")
end

function fit!(lale::LaleOp, xx::DataFrame, y::Vector=Vector())::Nothing
  x = xx |> Array
  impl_args = copy(lale.model[:impl_args])
  learner = lale.model[:learner]
  type = lale.model[:type]
  py_learner=PyObject
  if type == "sklearn"
     py_learner = getproperty(sk_dict[learner],learner)
  elseif type == "autogen"
     py_learner = getproperty(ag_dict[learner],learner)
  elseif type == "lale"
     py_learner = getproperty(ll_dict[learner],learner)
  else
     throw(ArgumentError("$learner not found"))
  end

  # Train
  modelobj = py_learner(;impl_args...)
  trained = PyNULL
  if :predict ∈ lalepropertynames(modelobj)
     # learner
     trained=modelobj.fit(x,y)
  else # transformer
     trained=modelobj.fit(x)
  end
  lale.model[:laleobj]   = trained
  lale.model[:impl_args] = impl_args
  return nothing
end

function fit(lale::LaleOp, xx::DataFrame, y::Vector=Vector())::LaleOp 
   fit!(lale,xx,y)
   lcopy =deepcopy(lale)
   return lcopy
end

function transform!(lale::LaleOp, xx::DataFrame)::Union{DataFrame, Vector}
   x = deepcopy(xx)|> Array
   laleobj = lale.model[:laleobj]
   # transform is predict for learners
   if :predict ∈ lalepropertynames(laleobj)
      return collect(laleobj.predict(x))
   elseif :transform ∈ lalepropertynames(laleobj)
      return collect(laleobj.transform(x)) |> x -> DataFrame(x,:auto)
   else
      throw(KeyError("predict/transform function not available"))
   end
end

function transform(lale::LaleOp, xx::DataFrame)::Union{DataFrame, Vector}
   return transform!(lale,xx)
end

predict(lale::LaleOp, xx::DataFrame)  = transform!(lale,xx)

function pretty_print(lop::LaleOp;args...)
~    lop.model[:laleobj].pretty_print(;args...)
end

function lalesummary(op::LaleOp)
   pyobject = op.model[:laleobj]
   if :summary ∈ lalepropertynames(pyobject)
      pyobject.summary()
   else
      nothing
   end
end


end
