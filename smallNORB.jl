module smallNORB
using Statistics
using Base.Iterators: repeated, partition
using Flux

using wgan: WGAN, trainWGAN, DCGANCritic, DCGANGenerator, MLPCritic, MLPGenerator
using FileIO, Images
# using CuArrays



norbImgSize = 96

function getsNORBImages(maxImages = 10000000)
    numTrainImages = 0
    imagePaths = Array{String}(undef, 0)
    for (root, dirs, files) in walkdir("./small_norb/smallnorb_export/train/")
        #println("Files in $root")
        for file in files
            push!(imagePaths, joinpath(root, file))
            numTrainImages += 1
            if numTrainImages == maxImages
                break
            end
            #println(joinpath(root, file)) # path to files
        end
    end
    @info("Example image path: $(imagePaths[1])...")
    return [load(imgPath) for imgPath in imagePaths]
end

# Bundle images together with labels and group into minibatchess

function makeMinibatch(X, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    return X_batch
end


    function make_minibatch_mlp(X, idxs)
    X_batch = Array{Float32}(undef, norbImgSize * norbImgSize, length(idxs))
    for i in 1:length(idxs)
        #print(X[idxs[i]])
        X_batch[:, i] = reshape(Float32.(X[idxs[i]]), :)
    end
    return X_batch
end


function DCGANCritic(useGPU::Bool = false)
    model = Chain(x->reshape(x, norbImgSize, norbImgSize, 1, :),
        Conv((4, 4), 1 => 32, stride = 2, relu),

        # Second convolution, operating upon a 14x14 image
        Conv((4, 4), 32 => 32, stride = 2),
        BatchNorm(32, relu),

        # Third convolution, operating upon a 7x7 image
        Conv((4, 4), 32 => 64, stride = 2),
        BatchNorm(64, relu),

        Conv((4, 4), 64 => 64, stride = 2),
        BatchNorm(64, relu),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x->reshape(x, :, size(x, 4)),
        Dense(1024, 1),
    )
    return DCGANCritic(model, useGPU)
end

function DCGANGenerator(;generatorInputSize = 10)
    model = Chain(Dense(generatorInputSize, 288),
        x->reshape(x, 3, 3, 32, :),
        ConvTranspose((3, 3), 32 => 32),
        BatchNorm(32, relu),

        # Second convolution, operating upon a 14x14 image
        ConvTranspose((3, 3), 32 => 16),
        BatchNorm(16, relu),

        # Third convolution, operating upon a 7x7 image
        ConvTranspose((3, 3), 16 => 1, σ),
    )
    return DCGANGenerator(model)
end



function MLPCritic()
    model = Chain(x->reshape(x, norbImgSize^2, :),
        Dense(norbImgSize^2, 264, relu),
        Dense(264, 1))
    return MLPCritic(model)
end

function MLPGenerator(useGPU::Bool = false; generatorInputSize = 100)
    model = Chain(
        Dense(generatorInputSize, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 512, relu),
        Dense(512, norbImgSize^2, σ),
        x->reshape(x, norbImgSize, norbImgSize, :))
    return MLPGenerator(model, useGPU)
end

function trainsNORBMLP()

    # Load labels and images from Flux.Data.sNORB
    @info("Loading data set...")
    train_imgs = getsNORBImages()

    batch_size = 32
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:

    @info("Constructing model...")
    wgan = WGAN(MLPCritic(), MLPGenerator(), generatorInputSize = 100)
   
    trainWGAN(wgan, train_set, train_set; modelName = "sNORB_mlp_large", numSamplesToSave = 40, imageSize = norbImgSize)

end

function trainsNORBMLPGeneratorDCGANCritic(; useGPU = false)

    # Load labels and images from Flux.Data.sNORB
    @info("Loading data set")
    train_imgs = getsNORBImages()

    batch_size = 64
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    generatorInputSize = 100
    @info("Constructing model...")
    wgan = WGAN(
        DCGANCritic(useGPU),
        MLPGenerator(useGPU, generatorInputSize = generatorInputSize),
        generatorInputSize = generatorInputSize,
        batchSize = batch_size,
        learningRate = 0.00005
    )
    
    if (useGPU) train_set = gpu.(train_set) end

    trainWGAN(wgan, train_set, train_set; modelName = "sNORB_mlp_dcgan_v2", numSamplesToSave = 40, imageSize = norbImgSize)

end
#getsNORBImages()
#trainsNORBMLP()
trainsNORBMLPGeneratorDCGANCritic()

end # module smallNORB
