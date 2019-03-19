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

Performs forward propagation on the input array `x`.
"""
(l::Connected)(x::AbstractArray) = l.σ.(l.W * x .+ l.b)

"""A convolutional NN layer"""
struct Convolution{F,A,V}
    W::A
    b::V
    σ::F
end

function Convolution(filter::Int, inCh::Int, outCh::Int, σ = identity; init = randn)
    # Weights will have dimensions (filter x filter x inCh x outCh)
    # i.e. there is a filter of weights for each input channel of each feature map.
    # Each feature map only has one bias weight.
    W = param(init(filter, filter, inCh, outCh))
    b = param(zeros(outCh))
    return Convolution(W, b, σ) 
end

Flux.@treelike Convolution

function (c::Convolution)(x::AbstractArray)
    # Initialize the pre-activated feature map values
    filterDim, _, inCh, outCh = size(c.W)
    xDimRows, xDimCols = size(x)[1:2]
    net = zeros(xDimRows - filterDim - 1, xDimCols - filterDim - 1, outCh)
    # Compute the nets
    convRows, convCols = size(net)[1:2]
    for featMap in 1:outCh
        for channel in 1:inCh
            # convolve over this input matrix
            inᵢ, inⱼ = 1, 1
            for netⱼ in 1:convCols
                for netᵢ in 1:convRows
                    # TODO: Start multiplying things.
                end
            end
        end
    end
    # add bias and activate
    return c.σ.(net .+ c.b)
end

end # module layers