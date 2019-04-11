module main
push!(LOAD_PATH, "./")
#include("mnist.jl")
#include("smallNORB.jl")
#include("mnist_mlp_dcgan/losses.jl")
end