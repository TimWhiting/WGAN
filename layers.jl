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
    filterDim::Int
    inCh::Int
    outCh::Int
end

function Convolution(filter::Int, inCh::Int, outCh::Int, σ = identity; init = randn)
    # Weights will have dimensions (filter x filter x inCh x outCh)
    # i.e. there is a filter of weights for each input channel of each feature map.
    # Each feature map only has one bias weight.
    W = param(init(filter, filter, inCh, outCh))
    b = param(zeros(outCh))
    return Convolution(W, b, σ, filter, inCh, outCh) 
end

Flux.@treelike Convolution

"""
Performs forward convolution on the input array `x`. `x` should be
in the HWC (height-width-channels) format.
"""
function (c::Convolution)(x::AbstractArray)
    # Initialize the pre-activated feature map values
    xDimRows, xDimCols = size(x)[1:2]
    net = Array{Any}(undef, xDimRows - c.filterDim + 1, xDimCols - c.filterDim + 1, c.outCh)
    # Compute the nets
    numRowSteps, numColSteps = size(net)[1:2]
    for featMap in 1:c.outCh
        # convolve over the input matrix to get
        # this feature map
        for netⱼ in 1:numColSteps, netᵢ in 1:numRowSteps
            xRows = netᵢ:(netᵢ + c.filterDim - 1)
            xCols = netⱼ:(netⱼ + c.filterDim - 1)
            chProducts = sum(sum(x[xRows,xCols, channel] .* c.W[:, :, channel, featMap]) for channel in 1:c.inCh)
            net[netᵢ, netⱼ, featMap] = chProducts
        end
        # Add the bias
        net[:,:,featMap] .+ c.b[featMap]
    end
    # Activate
    return c.σ.(net)
end

"""A convolutional transpose NN layer"""
struct ConvolutionTranspose{F,A,V}
    CTM::A # ctm stands for Convolution Transpose Matrix 
    b::V
    σ::F
    filterDim::Int
    inCh::Int
    outCh::Int
    xWidth::Int
    xHeight::Int
end

function ConvolutionTranspose(filter::Int, inCh::Int, outCh::Int, xWidth::Int, xHeight::Int, σ = identity; init = randn)
    # Final convolution transpose matrix (CTM) will have dimensions
    # (filter^2 x elementsInX x inCh x outCh).
    # Each feature map only has one bias weight.
    W = param(init(filter, filter, inCh, outCh))
    b = param(zeros(outCh))
    numColSteps = xWidth - filter + 1
    numRowSteps = xHeight - filter + 1
    # We initialize CTM to be of type Any so it can hold both floats untracked
    # by Flux's automatic differentiator AND the weights that are.
    CTM = Array{Any}(undef, numColSteps*numRowSteps, xWidth*xHeight, inCh, outCh)
    CTM[:] = zeros(size(CTM)...)
    CTMⱼ = 0
    for inChᵢ in 1:inCh, outChᵢ in 1:outCh
        for CTMᵢ in 1:size(CTM)[1]
            # Add a row to (CTM), the convolution transpose matrix
            # which will contain the weights (W) within it.
            CTMⱼ = Int(floor((CTMᵢ-1) / numColSteps) * xWidth) + (CTMᵢ-1) % numRowSteps + 1
            for Wᵢ in 1:filter
                # Insert row Wᵢ of W into row CTMᵢ of CTM at the proper place.
                CTM[CTMᵢ, CTMⱼ:(CTMⱼ+filter-1), inChᵢ, outChᵢ] = W[Wᵢ, :, inChᵢ, outChᵢ]
                CTMⱼ += xWidth
            end
        end
    end
    CTM = permutedims(CTM, [2,1,3,4])
    return ConvolutionTranspose(CTM, b, σ, filter, inCh, outCh, xWidth, xHeight) 
end

Flux.@treelike ConvolutionTranspose

"""
Performs forward convolution transpose on the input array
`x`. `x` should be in the HWC (height-width-channels) format.
"""
function (c::ConvolutionTranspose)(x::AbstractArray)
    if size(x)[1] != c.xHeight - c.filterDim + 1
        throw(ArgumentError("Incoming array `x` must have $(c.xHeight - c.filterDim + 1) rows, not $(size(x)[1])"))
    elseif size(x)[2] != c.xWidth - c.filterDim + 1
        throw(ArgumentError("Incoming array `x` must have $(c.xWidth - c.filterDim + 1) columns, not $(size(x)[2])"))
    end

    # flatten each channel into a vector
    flatX = reshape(permutedims(x, [2,1,3]), size(x)[1] * size(x)[2], size(x)[3])
    # Initialize the pre-activated feature map values
    net = Array{Any}(undef, c.xHeight, c.xWidth, c.outCh)
    # Compute the nets
    numRowSteps, numColSteps = size(net)[1:2]
    for featMap in 1:c.outCh
        # Calculate and sum the nets across all input channels
        chNets = sum(c.CTM[:, :, ch, featMap] * flatX[:, ch] for ch in 1:c.inCh)
        # Add the bias
        net[:, :, featMap] = reshape(chNets, size(net)[1], size(net)[2]) .+ c.b[featMap]
    end
    # Activate
    return c.σ.(net)
end

export Connected, Convolution, ConvolutionTranspose

end # module layers