### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b1ed3812-8cce-11eb-14ae-2f44e7ea24a2
begin
	using Lale
	using InteractiveUtils
    using Random
	using Statistics
	using Test
	using DataFrames: DataFrame
	using AutoMLPipeline: Utils
end

# ╔═╡ d27e91ae-8cce-11eb-1e68-cd8f37cb9a8b
begin
	iris = getiris()
	Xreg = iris[:,1:3] |> DataFrame
	Yreg = iris[:,4]   |> Vector
	Xcl  = iris[:,1:4] |> DataFrame
	Ycl  = iris[:,5]   |> Vector
end

# ╔═╡ c7756074-8cce-11eb-2698-91e4400a0b92
begin
	pca     = LalePreprocessor("PCA")
	rb      = LalePreprocessor("RobustScaler")
	noop    = LalePreprocessor("NoOp")
	rfr     = LaleLearner("RandomForestRegressor")
	rfc     = LaleLearner("RandomForestClassifier")
	treereg = LaleLearner("DecisionTreeRegressor")
	ohe  = OneHotEncoder()
	catf = CatFeatureSelector()
	numf = NumFeatureSelector()
end

# ╔═╡ 20cd74ea-8ccf-11eb-2f55-55aa24161f1e
begin
	# regression
	lalepipe =  (pca + noop) >>  (rfr | treereg )
	lale_hopt = LaleOptimizer(lalepipe,"Hyperopt",max_evals=10,cv=3)
	lalepred = fit_transform!(lale_hopt,Xreg,Yreg)
	lalermse=score(:rmse,lalepred,Yreg)
end

# ╔═╡ 3fa9bd06-8ccf-11eb-2d9d-159d5a7cc2b4
begin
	amlpipe = @pipeline  (pca + noop) |> (rfr * treereg)
	amlpred = fit_transform!(amlpipe,Xreg,Yreg)
	crossvalidate(amlpipe,Xreg,Yreg,"mean_squared_error")
	amlprmse=score(:rmse,amlpred,Yreg)
end

# ╔═╡ 52c60cfa-8ccf-11eb-3eaf-3dd8d8f59be1
begin
	# classification 
	lalepipec =  (pca) |> rfc
	lale_hoptc = LaleOptimizer(lalepipec,"Hyperopt",max_evals = 10,cv = 3)
	lalepredc  = fit_transform!(lale_hoptc,Xcl,Ycl)
	laleaccc   = score(:accuracy,lalepredc,Ycl)
end

# ╔═╡ 5d79e374-8ccf-11eb-0c48-03cb707018c6
begin
	amlpipec = @pipeline  (pca + rb) |> rfc
	amlpredc = fit_transform!(amlpipec,Xcl,Ycl)
	crossvalidate(amlpipec,Xcl,Ycl,"accuracy_score")
	amlpaccc = score(:accuracy,amlpredc,Ycl)
end

# ╔═╡ Cell order:
# ╠═b1ed3812-8cce-11eb-14ae-2f44e7ea24a2
# ╠═d27e91ae-8cce-11eb-1e68-cd8f37cb9a8b
# ╠═c7756074-8cce-11eb-2698-91e4400a0b92
# ╠═20cd74ea-8ccf-11eb-2f55-55aa24161f1e
# ╠═3fa9bd06-8ccf-11eb-2d9d-159d5a7cc2b4
# ╠═52c60cfa-8ccf-11eb-3eaf-3dd8d8f59be1
# ╠═5d79e374-8ccf-11eb-0c48-03cb707018c6
