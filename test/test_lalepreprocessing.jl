module TestLalePreprocessing

using Random
using Test
using Lale
using Statistics
using DataFrames: DataFrame, nrow

Random.seed!(1)

const IRIS = getiris()
extra = rand(150,3) |> x->DataFrame(x,:auto)
const X = hcat(IRIS[:,1:4],extra) 
const Y = IRIS[:,5] |> Vector

# "KernelCenterer","MissingIndicator","KBinsDiscretizer","OneHotEncoder", 
const preprocessors = [
     "FactorAnalysis", "FastICA", "IncrementalPCA",
     "KernelPCA", 
     "MiniBatchDictionaryLearning",
     "MiniBatchSparsePCA", "NMF", "PCA", 
     "TruncatedSVD", 
     "SimpleImputer",  
     "Binarizer",# "FunctionTransformer",
     "MaxAbsScaler", "MinMaxScaler", "Normalizer",
     "OrdinalEncoder", "PolynomialFeatures", "PowerTransformer", 
     "QuantileTransformer", "RobustScaler"#, #"StandardScaler"
 ]

function fit_test(preproc::String,in::DataFrame,out::Vector)
   _preproc=LaleOp(preproc,"autogen")
   fit!(_preproc,in,out)
   @test _preproc.model != Dict()
   return _preproc
end

function transform_test(preproc::String,in::DataFrame,out::Vector)
   _preproc=LaleOp(preproc,"autogen")
   fit!(_preproc,in,out)
   res = transform(_preproc,in)
   @test size(res)[1] == size(out)[1]
end

@testset "lale preprocessors fit test" begin
   Random.seed!(123)
   for cl in preprocessors
      fit_test(cl,X,Y)
   end
end

@testset "lale preprocessors transform test" begin
   Random.seed!(123)
   for cl in preprocessors
      transform_test(cl,X,Y)
   end
end

function skptest()
    features = X
    labels = Y

    pca = LaleOp("PCA","autogen",n_components=3)
    fit!(pca,features)
    @test transform(pca,features) |> x->size(x,2) == 3

    pca = LaleOp("PCA","autogen")
    fit!(pca,features)
    @test transform(pca,features) |> x->size(x,2) == 7

    svd = LaleOp("TruncatedSVD","autogen",n_components=2)
    fit!(svd,features)
    @test transform(svd,features) |> x->size(x,2) == 2

    ica = LaleOp("FastICA","autogen",n_components=2)
    fit!(ica,features)
    @test transform(ica,features) |> x->size(x,2) == 2

    stdsc = LaleOp("StandardScaler","autogen")
    fit!(stdsc,features)
    @test abs(mean(transform(stdsc,features) |> Matrix)) < 0.00001

    minmax = LaleOp("MinMaxScaler","autogen")
    fit!(minmax,features)
    @test mean(transform(minmax,features) |> Matrix) > 0.30

    vote = VoteEnsemble()
    stack = StackEnsemble()
    best = BestLearner()
    cat = CatFeatureSelector()
    num = NumFeatureSelector()
    disc = CatNumDiscriminator()
    ohe = OneHotEncoder()

    mpipeline = @pipeline (stdsc |> pca) |> best
    fit!(mpipeline,features,labels)
    pred = transform!(mpipeline,features)
    @test score(:accuracy,pred,labels) > 50.0

    fpipe = @pipeline ((cat + num) + (num + pca))  |> stack
    fit(fpipe,features,labels)
    @test ((transform(fpipe,features) .== labels) |> sum ) / nrow(features) > 0.50

end
@testset "scikit preprocessor fit/transform test with real data" begin
    Random.seed!(123)
    skptest()
end


end
