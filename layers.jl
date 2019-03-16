module layers

using Flux

"""A fully connected NN layer."""
struct Connected{F,S,T}
    W::S # These types need to remain fairly dynamic because
    b::T # it looks like Flux might change them around a little.
    σ::F
end

# We call param() on each thing we're training so Flux
# keeps track of computations that take place on those things,
# so it can perform backprop on them.
function Connected(inDim::Int, outDim::Int, σ::Function = identity; initW::Function = randn, initb::Function = randn)
    W = param(initW(outDim, inDim))
    b = param(initb(outDim))
    return Connected(W, b, σ)
end

# This I believe enables all the things param() has
# been called on within the layer to be collected
# automatically just by calling params() on the model or
# the layer as a whole, rather than having to pass each
# collected item explicitly to params().
Flux.@treelike Connected

"""
Allows an instantiated `Connected` layer to be called as a function e.g.

```julia
myLayer = Connected(5,10,σ)
x = [1,2,3,4,5]
h = myLayer(x) # performs Wx + b
```
"""
(l::Connected)(x::AbstractArray) = l.σ.(l.W * x .+ l.b)

"""A convolutional NN layer"""
struct Convolution{F,A,V}
    W::A
    b::V
    σ::F
end

function Convolution(filter::NTuple{N,Int}, ch::Pair{Int,Int}, σ = identity; init = randn)
    # Weights will have dimensions (filterDim1 x filterDim2 x numInputChannels x numOutputChannels)
    # i.e. there is a filter of weights for each input channel of each feature map.
    # Each feature map only has one bias weight.
    W = param(init(filter..., ch...))
    b = param(zeros(ch[2]))
    return Convolution(W, b, σ) 
end

Flux.@treelike Convolution

function (c::Convolution)(x::AbstractArray)
    # TODO
end

end # module layers