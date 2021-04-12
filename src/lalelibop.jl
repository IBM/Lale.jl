module LaleLibOps

using PyCall
import Pandas 

using DataFrames 
using Random
using ..AbsTypes
using ..Utils
using ..LaleAbsTypes

import ..AbsTypes: fit!, transform!
import ..LaleAbsTypes: fit, transform, predict

export fit!, transform!
export fit, transform, predict

import Base: >>,|,+,|>,&
export  >>,|,+,|>,&

export LalePipeOptimizer, lalepipeoptimizers
export LalePipe, visualize

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
   global _make_union_nc      = LALEOPS.make_union_no_concat
end

# hide pyobject
struct LalePipe <: LaleOperator
   model::Dict{Symbol, Any}
   LalePipe(x::PyObject) = new(Dict{Symbol,Any}(:laleobj=>x))
end

⊖(a::PyObject,b::PyObject)          = _make_pipeline(a,b) 
|>(a::LaleOperator,b::LaleOperator) = a.model[:laleobj] ⊖ b.model[:laleobj] |> LalePipe
|>(a::PyObject,b::LaleOperator)     = a ⊖ b.model[:laleobj] |> LalePipe
|>(a::LaleOperator,b::PyObject)     = a.model[:laleobj] ⊖ b |> LalePipe
>>(a::LaleOperator,b::LaleOperator) = a.model[:laleobj] ⊖ b.model[:laleobj] |> LalePipe
>>(a::PyObject,b::LaleOperator)     = a ⊖ b.model[:laleobj] |> LalePipe
>>(a::LaleOperator,b::PyObject)     = a.model[:laleobj] ⊖ b |> LalePipe

⊕(a::PyObject,b::PyObject)          = _make_union(a,b) 
+(a::LaleOperator,b::LaleOperator)  = a.model[:laleobj] ⊕ b.model[:laleobj] |> LalePipe
+(a::PyObject,b::LaleOperator)      = a ⊕ b.model[:laleobj] |> LalePipe
+(a::LaleOperator,b::PyObject)      = a.model[:laleobj] ⊕ b |> LalePipe

⨸(a::PyObject,b::PyObject)           = _make_union_nc(a,b) 
(&)(a::LaleOperator,b::LaleOperator) = a.model[:laleobj] ⨸ b.model[:laleobj] |> LalePipe
(&)(a::PyObject,b::LaleOperator)     = a ⨸ b.model[:laleobj] |> LalePipe
(&)(a::LaleOperator,b::PyObject)     = a.model[:laleobj] ⨸ b |> LalePipe

⊗(a::PyObject,b::PyObject)          = _make_choice(a,b) 
|(a::LaleOperator,b::LaleOperator)  = a.model[:laleobj] ⊗ b.model[:laleobj] |> LalePipe
|(a::PyObject,b::LaleOperator)      = a ⊗ b.model[:laleobj] |> LalePipe
|(a::LaleOperator,b::PyObject)      = a.model[:laleobj] ⊗ b |> LalePipe

mutable struct LalePipeOptimizer <: LaleOperator
   name::String
   model:: Dict{Symbol,Any}

   function LalePipeOptimizer(args=Dict{Symbol,Any}())
      default_args=Dict{Symbol,Any}(
            :lalepipe => PyNULL(),
            :name => "LalePipeOptimizer",
            :impl_args => Dict{Symbol,Any}(
                :cv        => 3,
                :max_evals => 10,
            )
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      @assert cargs[:lalepipe] != PyNULL()
      new(cargs[:name],cargs)
   end
end

function LalePipeOptimizer(pipe::LalePipe, args::Dict)
   LalePipeOptimizer(Dict(:lalepipe => pipe.model[:laleobj], args...))
end

function LalePipeOptimizer(pipe::LalePipe; args...)
   LalePipeOptimizer(Dict(:lalepipe => pipe.model[:laleobj], :impl_args => Dict(pairs(args))))
end

function lalepipeoptimizers()
  opts = keys(optim_dict) |> collect |> x -> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: LalePipeOptimizer(name::String, args::Dict=Dict())")
  println("where 'name can be one of:")
  println()
  [print(opt," ") for opt in opts]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Lale online help for more details about the auto_configure arguments.")
end

function fit!(lopt::LalePipeOptimizer, xx::DataFrame, y::Vector=Vector())
   Xpd = Pandas.DataFrame(xx).pyo
   Ypd = Pandas.DataFrame(y).pyo
   margs = lopt.model[:impl_args]
   pipe = lopt.model[:lalepipe]
   hyperopt = LALELIBS.Hyperopt(estimator=pipe; margs...)
   trained = hyperopt.fit(Xpd,Ypd)
   lopt.model[:trained] = trained
end

function transform!(lopt::LalePipeOptimizer, xx::DataFrame)
   Xpd = Pandas.DataFrame(xx).pyo
   trainedmodel = lopt.model[:trained]
   trainedmodel.predict(Xpd) |> Pandas.DataFrame |> DataFrame |> x -> x[:,1]
end

function fit(lopt::LalePipeOptimizer, xx::DataFrame, y::Vector=Vector()) 
   fit!(lopt,xx,y)
   loptcopy = deepcopy(lopt)
   return loptcopy
end

transform(lopt::LalePipeOptimizer, xx::DataFrame) = transform!(lopt,xx)
predict(lopt::LalePipeOptimizer, xx::DataFrame) = transform!(lopt,xx)

function visualize(lopt::LalePipeOptimizer)
   auto_trained = lopt.model[:trained]
   best_pipeline = auto_trained.get_pipeline()
   best_pipeline.visualize(ipython_display=false)
end

function visualize(lpipe::LalePipe)
   lpipe.model[:laleobj].visualize(ipython_display=false)
end

end
