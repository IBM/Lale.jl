### Lale.jl: Julia port of python's lale package

| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

---------
Lale.jl is a julia port of python's lale library for semi-automated data science. Lale makes it easy to automatically select algorithms and tune hyperparameters of pipelines that are compatible with scikit-learn, in a type-safe fashion.

Instructions for Lale developers can be found [here](./docs/DevInstruction.md)

### Package Features
- __automation__: provides a consistent high-level interface to existing pipeline search tools including Hyperopt, GridSearchCV, and SMAC
- __correctness checks__: uses JSON Schema to catch mistakes when there is a mismatch between hyperparameters and their type, or between data and operators
- __interoperability__: supports growing library of transformers and estimators

Here is an example of a typical lale pipeline using the following processing elements: principal 
component analysis (pca), noop (no operation), random forest regression (rf), 
and decision tree regression (tree):

```julia
lalepipe = (pca + noop) >> (rf | tree)
laleopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals = 10,cv = 3)
fit!(laleopt, Xtrain,Ytrain)
pred = transform!(laleopt,Xtest)
```
The block of code above will jointly search the optimal hyperparameters 
of both random forest and decision tree and select the best learner while at 
the same time search the optimal structure of the preprocessing elements, i.e., 
whether to use pca or not. The `>>` operator is used to
compose pipeline while the `+` operator is used to concatenate subpipelines.
Finally, the choice operator `|` signifies to the LaleOptimizer 
to search and select which among its pipeline elements will be optimal.

### Installation
Lale is in the Julia General package registry. The latest
release can be installed from the julia prompt:
```julia
julia> using Pkg
julia> Pkg.update()
julia> Pkg.add("Lale")
```
or use Julia's pkg shell which can be triggered by `]`
```julia
julia> ]
pkg> update
pkg> add Lale
```

#### Sample Lale Workflow
```julia
using Lale

using DataFrames
using AutoMLPipeline: Utils

# load data
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

# Lale regression
lalepipe  = (pca + noop) >>  (rfr | treereg )
lale_hopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals = 10,cv = 3)
lalepred  = fit_transform!(lale_hopt,Xreg,Yreg)
lalermse  = score(:rmse,lalepred,Yreg)

# Lale classification
lalepipe  = (rb + pca) |> rfc
lale_hopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals = 10,cv = 3)
lalepred  = fit_transform!(lale_hopt,Xcl,Ycl)
laleacc   = score(:accuracy,lalepred,Ycl)
```


[contrib-url]: https://github.com/IBM/Lale.jl/blob/main/CONTRIBUTORS.md
[issues-url]: https://github.com/IBM/Lale.jl/issues

[discourse-tag-url]: https://discourse.julialang.org/

[gitter-url]: https://gitter.im/AutoMLPipelineLearning/community
[gitter-img]: https://badges.gitter.im/ppalmes/TSML.jl.svg

[slack-img]: https://img.shields.io/badge/chat-on%20slack-yellow.svg
[slack-url]: https://julialang.slack.com/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ibm.github.io/Lale.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://ibm.github.io/Lale.jl/dev/

[travis-img]: https://travis-ci.com/IBM/Lale.jl.svg?branch=main
[travis-url]: https://travis-ci.com/IBM/Lale.jl

[codecov-img]: https://codecov.io/gh/IBM/Lale.jl/branch/main/graph/badge.svg?token=YK62W9KQ2T
[codecov-url]: https://codecov.io/gh/IBM/Lale.jl
