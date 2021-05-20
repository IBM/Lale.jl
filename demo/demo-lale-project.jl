using PyCall
using Pandas

sklearn = pyimport("sklearn")
pd = pyimport("pandas")
lsk = pyimport("lale.lib.sklearn")
llale = pyimport("lale.lib.lale")
Project = llale.Project
OneHotEncoder = lsk.OneHotEncoder
ConcatFeatures = llale.ConcatFeatures
Tree = lsk.DecisionTreeClassifier
KNN = lsk.KNeighborsClassifier
RF =  lsk.RandomForestClassifier
PCA = lsk.PCA
Hyperopt = llale.Hyperopt

train_test_split = pyimport("sklearn.model_selection").train_test_split
mean_squared_error = pyimport("sklearn.metrics").mean_squared_error
metrics = pyimport("sklearn.metrics")
accuracy_scorer = metrics.make_scorer(sklearn.metrics.accuracy_score)

df = Pandas.read_csv("credit.csv")
r,c = size(df)
y = df["class"].pyo
X = iloc(df)[1:r,2:c]

train_X, test_X, train_y, test_y = train_test_split(X, y)

prep_to_numbers = (
  (Project(columns=Dict("type"=>"string")) >> OneHotEncoder(handle_unknown = "ignore")) &
  Project(columns=Dict("type"=>"number")) )>> ConcatFeatures
planned_orig = prep_to_numbers >> ( Tree | KNN)

best_estimator = planned_orig.auto_configure(
    train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=10)
println("accuracy = $(accuracy_scorer(best_estimator, test_X, test_y))")

pca_tree_planned = prep_to_numbers >> PCA >> RF
pca_tree_trained = pca_tree_planned.auto_configure(
    train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=10, verbose=true)
println("accuracy = $(accuracy_scorer(pca_tree_trained, test_X, test_y))")
