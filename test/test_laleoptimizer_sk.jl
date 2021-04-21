module TestLalePipeOptimizerSK

using Random
using Test
using Lale
using Statistics
using DataFrames: DataFrame

const IRIS = getiris()
const X    = IRIS[:,1:3] |> DataFrame
const XC   = IRIS[:,1:4] |> DataFrame
const YC   = IRIS[:,5] |> Vector
const Y    = IRIS[:,4] |> Vector

function pipeline_test()

   # lale ops
   pca     = LaleOp("PCA")
   rb      = LaleOp("RobustScaler")
   rfr     = LaleOp("RandomForestRegressor")
   rfc     = laleoperator("RandomForestClassifier")
   treereg = LaleOp("DecisionTreeRegressor")
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

   amlpipe = @pipeline  (pca + noop) |> (rfr * treereg)
   amlpred = fit_transform!(amlpipe,X,Y)
   amlprmse=score(:rmse,amlpred,Y)

   # regression
   lalepipe   = (pca + noop) >>  (rfr | treereg )
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
   lalepipe  = (pca  >>  rfc)
   lale_hopt = LalePipeOptimizer(lalepipe,max_evals = 3,cv = 3)
   lalepred  = fit_transform!(lale_hopt,XC,YC)
   laleacc   = score(:accuracy,lalepred,YC)

   amlpipe = @pipeline  (pca + noop) |> (rfc)
   amlpred = fit_transform!(amlpipe,XC,YC)
   amlpacc = score(:accuracy,amlpred,YC)
   
   @test abs(amlpacc - laleacc) > 0.0

   lalepipe    = (pca  >>  rfc)
   lale_hopt   = LalePipeOptimizer(lalepipe,max_evals = 3,cv = 3)
   laletrained = fit(lale_hopt,XC,YC)
   lalepred    = predict(laletrained,XC)
   laleacc     = score(:accuracy,lalepred,YC)

   @test abs(amlpacc - laleacc) > 0.0
   
   plr = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfr
   plc = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> rfc
   perfreg(x,y) = score(:rmse,x,y)
   perfcl(x,y) = score(:accuracy,x,y)
   @test crossvalidate(plr,X,Y,perfreg).mean < 0.3
   @test crossvalidate(plc,XC,YC,perfcl).mean > 80.0

end
@testset "Lale Pipeline Optimizer" begin
   pipeline_test()
end

end
