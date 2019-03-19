module layers

using Flux

# Flux needs these methods to convert TrackedReals to Floats
# for faster matrix multiplies
Base.Float64(x::Flux.Tracker.TrackedReal{T}) where T <: Number = Float64(x.data)
Base.Float32(x::Flux.Tracker.TrackedReal{T}) where T <: Number = Float32(x.data)

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

"""
Performs forward convolution on the input array `x`. `x` should be
in the HWC (height-width-channels) format.
"""
function (c::Convolution)(x::AbstractArray)
    # Initialize the pre-activated feature map values
    filterDim, _, inCh, outCh = size(c.W)
    xDimRows, xDimCols = size(x)[1:2]
    net = Array{Any}(undef, xDimRows - filterDim + 1, xDimCols - filterDim + 1, outCh)
    # Compute the nets
    featMapRows, featMapCols = size(net)[1:2]
    for featMap in 1:outCh
        # convolve over the input matrix to get
        # this feature map
        for netⱼ in 1:featMapCols, netᵢ in 1:featMapRows
            xRows = netᵢ:netᵢ+filterDim-1
            xCols = netⱼ:netⱼ+filterDim-1
            chProducts = sum(sum(x[xRows,xCols, channel] .* c.W[:, :, channel, featMap]) for channel in 1:inCh)
            net[netᵢ, netⱼ, featMap] = chProducts
        end
        # Add the bias
        net[:,:,featMap] .+ c.b[featMap]
    end
    # Activate
    return c.σ.(net)
end

export Connected, Convolution

end # module layers