using Lale
using InteractiveUtils

using Random
using Statistics
using Test
using DataFrames: DataFrame
using AutoMLPipeline: Utils

function pauseme(msg)
  println(msg)
  println(">>> enter key to continue <<<")
  read(stdin,Char);
  nothing
end

function generateX()
   Random.seed!(123)
   iris = getiris()
   Xreg = iris[:,1:3] |> DataFrame
   Yreg = iris[:,4]   |> Vector
   Xcl  = iris[:,1:4] |> DataFrame
   Ycl  = iris[:,5]   |> Vector
   return (Xcl,Ycl)
end

function lale_demo()
  Xcl,Ycl=generateX()
  println("\n")
  println(">>>>>>>>>><<<<<<<<<<")
  println("Welcome to Lale demo")
  println(">>>>>>>>>><<<<<<<<<<")
  println("\n")
  pauseme("Let's start by loading the Iris plant dataset and show the first 5 rows.")
  println("X features:")  ;println(first(Xcl,5))
  println("\nY features:"); println(first(Ycl,5))
  pauseme("\nLet's define some Lale Operators")
  println("""
    julia code:
          pca     = laleoperator("PCA")
          rb      = laleoperator("RobustScaler","autogen")
          noop    = laleoperator("NoOp","lale")
          rfc     = laleoperator("RandomForestClassifier")
          treec   = laleoperator("DecisionTreeClassifier")
  """)
  pca     = laleoperator("PCA")
  rb      = laleoperator("RobustScaler","autogen")
  noop    = laleoperator("NoOp","lale")
  rfc     = laleoperator("RandomForestClassifier")
  treec   = laleoperator("DecisionTreeClassifier")
  pauseme("Let's build a simple Lale pipeline for classification and optimize it.")
  println("""
    julia code:
        lalepipe    = (rb + pca + noop) >>  (rfc | treec)
        lale_hopt   = LalePipeOptimizer(lalepipe,max_evals = 50,cv = 3)
        laletrained = fit(lale_hopt,Xcl,Ycl)
        lalepred    = transform(laletrained,Xcl)
        laleacc     = score(:accuracy,lalepred,Ycl)
        println("accuracy:",laleacc)
  """)
  lalepipe    = (rb + pca + noop) >>  (rfc | treec)
  lale_hopt   = LalePipeOptimizer(lalepipe,max_evals = 10,cv = 3)
  laletrained = fit(lale_hopt,Xcl,Ycl)
  lalepred    = transform(laletrained,Xcl)
  laleacc     = score(:accuracy,lalepred,Ycl)
  println("accuracy:",laleacc)
  nothing
end
lale_demo()


