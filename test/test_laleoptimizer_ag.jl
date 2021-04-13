module TestLalePipeOptimizerAG

using Random
using Test
using Lale
using Statistics
using DataFrames: DataFrame
using AutoMLPipeline.Utils

const IRIS = getiris()
const X    = IRIS[:,1:3] |> DataFrame
const XC   = IRIS[:,1:4] |> DataFrame
const YC   = IRIS[:,5] |> Vector
const Y    = IRIS[:,4] |> Vector


function pipeline_test()

   # lale ops
   pca     = LaleOp("PCA","autogen")
   rb      = LaleOp("RobustScaler","autogen")
   rfr     = LaleOp("RandomForestRegressor")
   rfc     = laleoperator("RandomForestClassifier")
   svr     = LaleOp("SVR")
   noop    = laleoperator("NoOp","lale")

   @test_throws ArgumentError LaleOp("PCAN")
   @test_throws ArgumentError LaleOp("RandomForestClassifers")

   # amlp ops
   ohe  = OneHotEncoder()
   catf = CatFeatureSelector()
   numf = NumFeatureSelector()

   pcatrained            = fit(pca,X)
   res                   = transform(pcatrained,X)
   @assert size(res,2)   == 3  

   amlpipe = @pipeline  (rb + pca + noop) |> (rfr * svr)
   amlpred = fit_transform!(amlpipe,X,Y)
   amlprmse=score(:rmse,amlpred,Y)

   # regression
   lalepipe   = (rb + pca + noop) >>  (rfr | svr )
   lale_hopt  = LalePipeOptimizer(lalepipe,max_evals  = 3,cv = 3)
   lalepred   = fit_transform!(lale_hopt,X,Y)
   lalermse   = score(:rmse,lalepred,Y)

   @test amlprmse < lalermse

   laletrained = fit(lale_hopt,X,Y)
   lalepred    = transform(laletrained,X)
   lalermse    = score(:rmse,lalepred,Y)

   @test amlprmse < lalermse

   laletrained = fit(lale_hopt,X,Y)
   lalepred    = predict(laletrained,X)
   lalermse    = score(:rmse,lalepred,Y)

   @test amlprmse < lalermse
   
   # classification 
   lalepipe  = (rb+pca+noop)  >>  rfc
   lale_hopt = LalePipeOptimizer(lalepipe,max_evals = 3,cv = 3)
   lalepred  = fit_transform!(lale_hopt,XC,YC)
   laleacc   = score(:accuracy,lalepred,YC)

   amlpipe = @pipeline  (pca + noop) |> (rfc)
   amlpred = fit_transform!(amlpipe,XC,YC)
   amlpacc = score(:accuracy,amlpred,YC)
   
   @test abs(amlpacc - laleacc) < 50.0

   lalepipe    = (pca  >>  rfc)
   lale_hopt   = LalePipeOptimizer(lalepipe,max_evals = 3,cv = 3)
   laletrained = fit(lale_hopt,XC,YC)
   lalepred    = predict(laletrained,XC)
   laleacc     = score(:accuracy,lalepred,YC)

   @test abs(amlpacc - laleacc) < 50.0
   
   plr = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfr
   plc = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfc
   @test crossvalidate(plr,X,Y,"mean_absolute_error",3,false).mean < 0.3
   @test crossvalidate(plc,XC,YC,"accuracy_score",3,false).mean > 0.8

end
@testset "Lale Pipeline Optimizer" begin
   pipeline_test()
end

end
