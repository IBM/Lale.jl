### Lale.jl: Julia wrapper of python's lale package

| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

---------
Lale.jl is a Julia wrapper of Python's [Lale](https://github.com/ibm/lale) library for semi-automated data science. Lale makes it easy to automatically select algorithms and tune hyperparameters of pipelines that are compatible with scikit-learn, in a type-safe fashion.

Instructions for Lale developers can be found [here](./docs/DevInstruction.md).

For a quick notebook demo: [Lale Notebook Demo](./demo/demo-lale-package-notebook.ipynb) or you can view it with
[NBViewer](https://nbviewer.jupyter.org/github/IBM/Lale.jl/blob/main/demo/demo-lale-package-notebook.ipynb).

### Package Features
- __automation__: provides a consistent high-level interface to existing pipeline search tools including Hyperopt, GridSearchCV, and SMAC
- __correctness checks__: uses JSON Schema to catch mistakes when there is a mismatch between hyperparameters and their type, or between data and operators
- __interoperability__: supports growing library of transformers and estimators

Here is an example of a typical `Lale` pipeline using the following processing elements: Principal 
Component Analysis (PCA), NoOp (no operation), Random Forest Regression (RFR), 
and Decision Tree Regression (DTree):

```julia
lalepipe  = (PCA + NoOp) >> (RFR | DTree)
laleopt   = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
laletr    = fit!(laleopt, Xtrain,Ytrain)
pred      = transform!(laletr,Xtest)
```
The block of code above will jointly search the optimal hyperparameters 
of both Random Forest and Decision Tree learners and select the best 
learner while at the same time searching the optimal hyperparameters
of the PCA. 

The *pipe combinator*, `p1 >> p2`, first runs sub-pipeline
`p1` and then pipes its output into sub-pipeline `p2`.
The *union combinator*, `p1 + p2`, runs sub-pipelines `p1` and `p2` separately
over the same data, and then concatenates the output columns of both.
The *or combinator*, `p1 | p2`, creates an algorithmic choice for the optimizer
to search and select which between `p1` and `p2` yields better results.

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
pca     = laleoperator("PCA")
rb      = laleoperator("RobustScaler","autogen")
noop    = laleoperator("NoOp","lale")
rfr     = laleoperator("RandomForestRegressor")
rfc     = laleoperator("RandomForestClassifier")
treereg = laleoperator("DecisionTreeRegressor")

# Lale regression
lalepipe  = (pca + noop) >>  (rfr | treereg )
lale_hopt = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
laletrain = fit(lale_hopt,Xreg,Yreg)
lalepred  = transform(laletrain,Xreg)
lalermse  = score(:rmse,lalepred,Yreg)

# Lale classification
lalepipe  = (rb + pca) |> rfc
lale_hopt = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
laletrain = fit(lale_hopt,Xcl,Ycl)
lalepred  = transform(laletrain,Xcl)
laleacc   = score(:accuracy,lalepred,Ycl)
```


[contrib-url]: https://github.com/IBM/Lale.jl/blob/main/CONTRIBUTORS.md
[issues-url]: https://github.com/IBM/Lale.jl/issues

[discourse-tag-url]: https://discourse.julialang.org/

[gitter-url]: https://gitter.im/Lale/community
[gitter-img]: https://badges.gitter.im/ppalmes/Lale.jl.svg

[slack-img]: https://img.shields.io/badge/chat-on%20slack-yellow.svg
[slack-url]: https://julialang.slack.com/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ibm.github.io/Lale.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://ibm.github.io/Lale.jl/dev/

[travis-img]: https://github.com/IBM/Lale.jl/actions/workflows/ci.yml/badge.svg
[travis-url]: https://github.com/IBM/Lale.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/IBM/Lale.jl/branch/main/graph/badge.svg?token=YK62W9KQ2T
[codecov-url]: https://codecov.io/gh/IBM/Lale.jl
