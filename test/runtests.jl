module TestLale
using Test

# suppress warnings
@info "suppressing PyCall warnings"
using PyCall
warnings = pyimport("warnings")
warnings.filterwarnings("ignore")

# test modules
include("test_lalelearner.jl")
include("test_lalepreprocessing.jl")
include("test_laleoptimizer.jl")

end