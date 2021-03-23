# load julia packages
using PyCall
using Pandas

# python import
sk       = pyimport("sklearn")
pd       = pyimport("pandas")
np       = pyimport("numpy")
dt       = pyimport("lale.datasets")
pickle   = pyimport("pickle")
lale     = pyimport("lale")
RF       = pyimport("lale.lib.sklearn").RandomForestRegressor
LinReg   = pyimport("lale.lib.sklearn").LinearRegression
PCA      = pyimport("lale.lib.sklearn").PCA
ICA      = pyimport("lale.lib.autogen").FastICA
Tree     = pyimport("lale.lib.sklearn").DecisionTreeRegressor
Pipeline = pyimport("lale.operators").Pipeline

# @pyimport lale.lib.lale as lb
lb             = pyimport("lale.lib.lale")
Hyperopt       = lb.Hyperopt
GridSearchCV   = lb.GridSearchCV
NoOp           = lb.NoOp
ConcatFeatures = lb.ConcatFeatures

# @pyimport lale.operators as lops
laleops       = pyimport("lale.operators")
make_pipeline = laleops.make_pipeline
make_choice   = laleops.make_choice
make_union    = laleops.make_union

# define lale operators
import Base: >>, |,&, + 
#&(a::PyObject,b::PyObject)  = make_union(a,b) 
>>(a::PyObject,b::PyObject) = make_pipeline(a,b)
+(a::PyObject,b::PyObject)  = make_union(a,b)
|(a::PyObject,b::PyObject)  = make_choice(a,b) # UTF \times

# load data
(train_X, train_y), (test_X, test_y) = dt.california_housing_df()
pd.concat([train_X.head(), train_y.head()], axis=1)


# pipeline 1
pca_tree_planned = Pipeline(steps=[("tfm", PCA()), ("estim", Tree())])
pca_tree_planned.fit(train_X, train_y)
predicted = pca_tree_planned.predict(test_X);
print("R2 score: ",sk.metrics.r2_score(test_y, predicted))

# pipeline 2
pipeline    = PCA >> Tree
trainedhopt = pipeline.auto_configure(train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=10, verbose=true)
predicted   = trainedhopt.predict(test_X);
print("R2 score: ",sk.metrics.r2_score(test_y, predicted))

# pipeline 3
planned    = (PCA + NoOp)   >>  (LinReg | Tree)
trained    = planned.auto_configure(train_X, train_y, optimizer=Hyperopt, cv=3, max_evals=3, verbose=true)
predicted  = trained.predict(test_X);
print("R2 score: ",sk.metrics.r2_score(test_y, predicted))

# pipeline 4
planned    = (PCA(svd_solver="auto") + NoOp) >>  (LinReg | Tree)
trained    = planned.auto_configure(train_X, train_y, optimizer=Hyperopt, cv=3,  max_evals=10,verbose=true)
predicted  = trained.predict(test_X);
print("R2 score: ",sk.metrics.r2_score(test_y, predicted))

# pipeline 5
pipeline    = PCA(svd_solver="auto") >> Tree
trainedgrid = pipeline.auto_configure(train_X, train_y, optimizer=Hyperopt,max_evals=5, cv=3,verbose=true)
predicted   = trainedgrid.predict(test_X);
print("R2 score: ",sk.metrics.r2_score(test_y, predicted))


# still in experimental stage
#=
using Distributed
nprocs() == 1 && addprocs(5;exeflags="--project")

@everywhere using PyCall
@everywhere np            = pyimport("numpy")
@everywhere sk            = pyimport("sklearn")
@everywhere pd            = pyimport("pandas")
@everywhere dt            = pyimport("lale.datasets")
@everywhere lale          = pyimport("lale")
@everywhere make_pipeline = pyimport("lale.operators").make_pipeline
@everywhere make_union    = pyimport("lale.operators").make_union
@everywhere make_choice   = pyimport("lale.operators").make_choice
@everywhere Hyperopt      = pyimport("lale.lib.lale").Hyperopt
@everywhere NoOp          = pyimport("lale.lib.lale").NoOp
@everywhere RF            = pyimport("lale.lib.sklearn").RandomForestRegressor
@everywhere LinReg        = pyimport("lale.lib.sklearn").LinearRegression
@everywhere PCA           = pyimport("lale.lib.sklearn").PCA
@everywhere ICA           = pyimport("lale.lib.autogen").FastICA
@everywhere Tree          = pyimport("lale.lib.sklearn").DecisionTreeRegressor
@everywhere (train_X, train_y), (test_X, test_y) = dt.california_housing_df()
@everywhere import Base: >>, |,+
@everywhere >>(a::PyObject,b::PyObject) = make_pipeline(a,b)
@everywhere +(a::PyObject,b::PyObject)  = make_union(a,b)  # UTF \oplus
@everywhere |(a::PyObject,b::PyObject)  = make_choice(a,b) # UTF \times

@everywhere pipelines=[PCA(svd_solver="auto") >> Tree() , (PCA(svd_solver="auto") + NoOp()) >>  (LinReg() | Tree()) , (ICA() + NoOp()) >> LinReg() , PCA() >> LinReg() , ICA() >> RF()]

res = @sync  @distributed (vcat) for pipeline in pipelines
   trainedpipe = pipeline.auto_configure(train_X, train_y,optimizer=Hyperopt, cv=3, max_evals=3,verbose=true)
   predicted = trainedpipe.predict(test_X)
   sk.metrics.r2_score(test_y, predicted)
end
=#
