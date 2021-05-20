module TestImb

using Lale
using DataFrames
using PyCall
using Test

const Hyperopt = laleoperator("Hyperopt","lale")

function testpipe(pipe,train_X,train_y,test_X,test_y)
   optimizer = Hyperopt(pipe, max_evals = 10, scoring="roc_auc")
   trained_optimizer = fit(optimizer,train_X, train_y)
   predictions = predict(trained_optimizer,test_X)
   lalesummary(trained_optimizer)
   @test score(:accuracy,test_y,predictions) > 0.50
end

function imb_test()
   fetch = pyimport("lale.datasets.openml").fetch
   (train_X, train_y), (test_X, test_y) = fetch("breast-cancer", "classification", preprocess=true)
   train_X = DataFrame(train_X,:auto)
   test_X = DataFrame(test_X,:auto)

   MinMaxScaler = laleoperator("MinMaxScaler")
   PCA = laleoperator("PCA")
   NoOp = laleoperator("NoOp","lale")
   LR= laleoperator("LogisticRegression")
   RF= laleoperator("RandomForestClassifier")
   SMOTE = laleoperator("SMOTE","imblearn")
   ADASYN = laleoperator("ADASYN","imblearn")
   CondensedNearestNeighbour = laleoperator("CondensedNearestNeighbour","imblearn")
   SMOTEENN = laleoperator("SMOTEENN","imblearn")
   accuracy_score=pyimport("sklearn.metrics").accuracy_score

   basepipeline =  MinMaxScaler() >> PCA() >> RF()
   testpipe(basepipeline,train_X,train_y,test_X,test_y)
   
   # SMOTE
   correctedpipeline =  SMOTE(basepipeline)
   testpipe(correctedpipeline,train_X,train_y,test_X,test_y)

   # CondensedNearestNeighbour
   correctedpipeline =  CondensedNearestNeighbour(basepipeline)
   testpipe(correctedpipeline,train_X,train_y,test_X,test_y)

   # SMOTEENN
   correctedpipeline =  SMOTEENN(basepipeline)
   testpipe(correctedpipeline,train_X,train_y,test_X,test_y)

   # ADASYN
   correctedpipeline =  ADASYN(basepipeline)
   testpipe(correctedpipeline,train_X,train_y,test_X,test_y)

end
@testset "Imbalanced Dataset" begin
   imb_test()
end

end
