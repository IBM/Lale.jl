module LaleLibOps

using PyCall
import Pandas 

using DataFrames 
using Random
using ..AbsTypes
using ..Utils
using ..LaleAbsTypes

import ..AbsTypes: fit!, transform!
import ..LaleAbsTypes: fit, transform

export fit!, transform!
export fit, transform

import Base: >>,|,+,|>,&
export  >>,|,+,|>,&

export LaleOptimizer, laleoptimizers

const optim_dict    = Dict{String, PyObject}()
const LALELIBS      = PyNULL()
const LALEOPS       = PyNULL()

function __init__()
   copy!(LALELIBS, pyimport("lale.lib.lale"))
   copy!(LALEOPS,  pyimport("lale.operators"))

   optim_dict["Hyperopt"]     = LALELIBS.Hyperopt
   optim_dict["GridSearchCV"] = LALELIBS.GridSearchCV
   global _ConcatFeatures     = LALELIBS.ConcatFeatures
   global _make_pipeline      = LALEOPS.make_pipeline
   global _make_choice        = LALEOPS.make_choice
   global _make_union         = LALEOPS.make_union
end

⊖(a::PyObject,b::PyObject)         = _make_pipeline(a,b)
|>(a::LaleOperator,b::LaleOperator) = a.model[:laleobj] ⊖ b.model[:laleobj]
|>(a::PyObject,b::LaleOperator)     = a ⊖ b.model[:laleobj]
|>(a::LaleOperator,b::PyObject)     = a.model[:laleobj] ⊖ b
>>(a::LaleOperator,b::LaleOperator) = a.model[:laleobj] ⊖ b.model[:laleobj]
>>(a::PyObject,b::LaleOperator)     = a ⊖ b.model[:laleobj]
>>(a::LaleOperator,b::PyObject)     = a.model[:laleobj] ⊖ b

⊕(a::PyObject,b::PyObject)          = _make_union(a,b)
+(a::LaleOperator,b::LaleOperator)  = a.model[:laleobj] ⊕ b.model[:laleobj]
+(a::PyObject,b::LaleOperator)      = a ⊕ b.model[:laleobj]
+(a::LaleOperator,b::PyObject)      = a.model[:laleobj] ⊕ b
(&)(a::LaleOperator,b::LaleOperator)  = a.model[:laleobj] ⊕ b.model[:laleobj]
(&)(a::PyObject,b::LaleOperator)      = a ⊕ b.model[:laleobj]
(&)(a::LaleOperator,b::PyObject)      = a.model[:laleobj] ⊕ b

⊗(a::PyObject,b::PyObject)          = _make_choice(a,b)
|(a::LaleOperator,b::LaleOperator)  = a.model[:laleobj] ⊗ b.model[:laleobj]
|(a::PyObject,b::LaleOperator)      = a ⊗ b.model[:laleobj]
|(a::LaleOperator,b::PyObject)      = a.model[:laleobj] ⊗ b

mutable struct LaleOptimizer <: LaleOperator
   name::String
   model:: Dict{Symbol,Any}

   function LaleOptimizer(args=Dict{Symbol,Any}())
      default_args=Dict{Symbol,Any}(
            :lalepipe => PyNULL(),
            :name => "laleoptimizer",
            :optimizer => "Hyperopt",
            :impl_args => Dict{Symbol,Any}(
                :cv        => 3,
                :max_evals => 10,
                :verbose   => true
            )
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      opt = cargs[:optimizer]
      if !(opt in keys(optim_dict))
         println("$opt is not supported.")
         println()
         laleoptimizers()
         error("Argument keyword error")
      end
      @assert cargs[:lalepipe] != PyNULL()
      new(cargs[:name],cargs)
   end
end

function LaleOptimizer(pipe::PyObject, args::Dict)
   LaleOptimizer(Dict(:lalepipe => pipe, args...))
end

function LaleOptimizer(pipe::PyObject,optimizer::String="Hyperopt"; args...)
   LaleOptimizer(Dict(:lalepipe => pipe, :optimizer=>optimizer, :impl_args => Dict(pairs(args))))
end

function laleoptimizers()
  opts = keys(optim_dict) |> collect |> x -> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: LaleOptimizer(name::String, args::Dict=Dict())")
  println("where 'name can be one of:")
  println()
  [print(opt," ") for opt in opts]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Lale online help for more details about the auto_configure arguments.")
end

function fit!(lopt::LaleOptimizer, xx::DataFrame, y::Vector)
   Xpd = Pandas.DataFrame(xx).pyo
   Ypd = Pandas.DataFrame(y).pyo
   margs = lopt.model[:impl_args]
   optim = lopt.model[:optimizer]
   pipe = lopt.model[:lalepipe]
   trained = pipe.auto_configure(Xpd, Ypd, optimizer=optim_dict[optim]; margs...)
   lopt.model[:trained] = trained
end

function transform!(lopt::LaleOptimizer, xx::DataFrame)
   Xpd = Pandas.DataFrame(xx).pyo
   trainedmodel = lopt.model[:trained]
   trainedmodel.predict(Xpd) |> Pandas.DataFrame |> DataFrame |> x -> x[:,1]
end

function fit(lopt::LaleOptimizer, xx::DataFrame, y::Vector)
   Xpd = Pandas.DataFrame(xx).pyo
   Ypd = Pandas.DataFrame(y).pyo
   margs = lopt.model[:impl_args]
   optim = lopt.model[:optimizer]
   pipe = lopt.model[:lalepipe]
   trained = pipe.auto_configure(Xpd, Ypd, optimizer=optim_dict[optim]; margs...)
   lopt.model[:trained] = trained
end

function transform(lopt::LaleOptimizer, xx::DataFrame)
   Xpd = Pandas.DataFrame(xx).pyo
   trainedmodel = lopt.model[:trained]
   #trainedmodel.predict(Xpd) |> Pandas.DataFrame |> DataFrame |> x -> x[:,1]
   trainedmodel.predict(Xpd) |> x -> x[:,1]
end

end
