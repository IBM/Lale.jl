### Lale.jl: Julia wrapper of python's lale package

| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

---------
Lale.jl is a Julia wrapper of Python's [Lale](https://github.com/ibm/lale) library for semi-automated data science. Lale makes it easy to automatically select algorithms and tune hyperparameters of pipelines that are compatible with scikit-learn, in a type-safe fashion.

More details of Lale: [AutoML@KDD](https://arxiv.org/abs/2007.01977)

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
using DataFrames: DataFrame

# load data
iris = getiris()
Xreg = iris[:,1:3] |> DataFrame
Yreg = iris[:,4]   |> Vector
Xcl  = iris[:,1:4] |> DataFrame
Ycl  = iris[:,5]   |> Vector

# regression dataset
regsplit = train_test_split(Xreg,Yreg;testprop = 0.20)
trXreg,trYreg,tstXreg,tstYreg = regsplit

# classification dataset
clsplit = train_test_split(Xcl,Ycl;testprop = 0.20)
trXcl,trYcl,tstXcl,tstYcl = clsplit

# lale ops
pca     = laleoperator("PCA")
rb      = laleoperator("RobustScaler")
noop    = laleoperator("NoOp","lale")
rfr     = laleoperator("RandomForestRegressor")
rfc     = laleoperator("RandomForestClassifier")
treereg = laleoperator("DecisionTreeRegressor")

# Lale regression
lalepipe  = (pca + noop) >>  (rfr | treereg )
lale_hopt = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
laletrain = fit(lale_hopt,trXreg,trYreg)
lalepred  = transform(laletrain,tstXreg)
score(:rmse,lalepred,tstYreg) |> println

# Lale classification
lalepipe  = (rb + pca) |> rfc
lale_hopt = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
laletrain = fit(lale_hopt,trXcl,trYcl)
lalepred  = transform(laletrain,tstXcl)
score(:accuracy,lalepred,tstYcl) |> println
```
Moreover, Lale is also compatible with [AutoMLPipeline](https://github.com/IBM/AutoMLPipeline.jl) `@pipeline` syntax:
```julia
# regression pipeline
regpipe      = @pipeline (pca + rb) |>  rfr
regmodel     = fit(regpipe,trXreg, trYreg)
regpred      = transform(regmodel,tstXreg)
regperf(x,y) = score(:rmse,x,y)
regperf(regpred, tstYreg) |> println
crossvalidate(regpipe,Xreg,Yreg,regperf)

# classification pipeline
clpipe         = @pipeline (pca + noop) |>  rfc
clmodel        = fit(clpipe,trXcl, trYcl)
clpred         = transform(clmodel,tstXcl)
classperf(x,y) = score(:accuracy,x,y)
classperf(clpred, tstYcl) |> println
crossvalidate(clpipe,Xcl,Ycl,classperf)
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
