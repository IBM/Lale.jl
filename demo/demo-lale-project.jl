using Lale
using DataFrames
using CSV

strfeat= laleoperator("Project","lale")(columns=Dict("type"=>"string"))
numfeat = laleoperator("Project","lale")(columns=Dict("type"=>"number"))

OneHotEncoder = laleoperator("OneHotEncoder")
ConcatFeatures = laleoperator("ConcatFeatures","lale")
Tree = laleoperator("DecisionTreeClassifier")
KNN = laleoperator("KNeighborsClassifier")
RF =  laleoperator("RandomForestClassifier")
PCA = laleoperator("PCA")
Hyperopt = laleoperator("Hyperopt","lale")

df = CSV.read("./demo/old/credit.csv",DataFrame) 
y = df[!,"class"] |> collect
X = df[:,2:end]

train_X,train_y,test_X,test_y = Lale.train_test_split(X, y,testprop=0.20)

prep_to_numbers = 
 ((strfeat >> OneHotEncoder(handle_unknown = "ignore")) & numfeat)>> ConcatFeatures
planned_orig = prep_to_numbers >> ( Tree | KNN)
lopt = LalePipeOptimizer(planned_orig,max_evals = 10,cv = 3)
laletrained = fit(lopt,train_X,train_y)
lalepred  = Lale.transform(laletrained,test_X)
score(:accuracy,lalepred,test_y) |> println

pca_tree_planned = prep_to_numbers >> PCA >> RF
lopt = LalePipeOptimizer(pca_tree_planned,max_evals = 10,cv = 3)
laletrained = fit(lopt,train_X,train_y)
lalepred  = Lale.transform(laletrained,test_X)
score(:accuracy,lalepred,test_y) |> println
