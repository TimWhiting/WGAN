module datasets

using Random

"""
Makes data with two classes.
Makes it as one big randomly shuffled minibatch.
Example shape (an unfilled square):

Square:     Box:
0 0 0 0 0   0 0 0 0 0
0 1 1 1 0   0 1 1 1 0
0 1 0 1 0   0 1 1 1 0
0 1 1 1 0   0 1 1 1 0
0 0 0 0 0   0 0 0 0 0

For the labels, [1; 0] means its a square, and [0; 1] means its a box.

TODO: Make able to be inverted for the convolution
transpose case i.e. where the data is the target.
"""
function makeTwoClassShapes(boxDims::Tuple{Int64,Int64} = (5,5); isInvert::Bool = false)
    any(x -> x < 3, boxDims) && throw(ArgumentError("each dimension of `boxDims` must be `>= 3`"))
    r, c = boxDims
    rSteps = r - 3 + 1
    cSteps = c - 3 + 1
    n = rSteps * cSteps * 2
    data = zeros(Float32, r, c, 1, n)
    labels = zeros(Float32, 2, n)

    # Make the squares
    i = 0
    dataIndices = shuffle(1:n)
    idx = 1
    for sᵢ in 1:rSteps, sⱼ in 1:cSteps
        i += 1
        idx = dataIndices[i]
        data[sᵢ, sⱼ:sⱼ+2, 1, idx] = [1. 1. 1.]
        data[sᵢ+1, sⱼ:sⱼ+2, 1, idx] = [1. 0. 1.]
        data[sᵢ+2, sⱼ:sⱼ+2, 1, idx] = [1. 1. 1.]
        # Add the label
        labels[1, idx] = 1
    end

    # make the boxes
    for bᵢ in 1:rSteps, bⱼ in 1:cSteps
        i += 1
        idx = dataIndices[i]
        data[bᵢ:bᵢ+2, bⱼ:bⱼ+2, 1, idx] = [1. 1. 1.; 1. 1. 1.; 1. 1. 1.;]
        # Add the label
        labels[2, idx] = 1
    end
    
    return (data, labels)
end

export makeTwoClassShapes

end # module datasets