module smallNORB
using Statistics
using Base.Iterators: repeated, partition
using Flux

using wgan: WGAN, trainWGAN, DCGANCritic, DCGANGenerator, MLPCritic, MLPGenerator
using FileIO, Images


norbImgSize = 96

function getsNORBImages()
    numTrainImages = 0
    imagePaths = Array{String}(undef, 0)
    for (root, dirs, files) in walkdir("./smallnorb_export/train/")
        #println("Files in $root")
        for file in files
            push!(imagePaths, joinpath(root, file))
            #println(joinpath(root, file)) # path to files
        end
    end
    print(imagePaths[1])
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


function DCGANCritic()
    model = Chain(x->reshape(x, norbImgSize, norbImgSize, 1, :),
        Conv((3, 3), 1 => 16, pad = (1, 1)),
        BatchNorm(16, relu),
        x->maxpool(x, (2, 2)),

        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16 => 32, pad = (1, 1)),
        BatchNorm(32, relu),
        x->maxpool(x, (2, 2)),

        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32 => 32, pad = (1, 1)),
        BatchNorm(32, relu),
        x->maxpool(x, (2, 2)),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x->reshape(x, :, size(x, 4)),
        Dense(288, 1),
    )
    return DCGANCritic(model)
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
        Dense(norbImgSize^2, 128, relu),
        Dense(128, 1))
    return MLPCritic(model)
end

function MLPGenerator(;generatorInputSize = 10)
    model = Chain(Dense(generatorInputSize, 128, relu),
        Dense(128, norbImgSize^2, σ),
        x->reshape(x, norbImgSize, norbImgSize, :))
    return MLPGenerator(model)
end

function trainsNORBMLP()

    # Load labels and images from Flux.Data.sNORB
    @info("Loading data set")
    train_imgs = getsNORBImages()

    batch_size = 32
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:

    @info("Constructing model...")
    wgan = WGAN(MLPCritic(), MLPGenerator())
   
    trainWGAN(wgan, train_set, train_set; modelName = "sNORB_mlp", numSamplesToSave = 40, imageSize = norbImgSize)

end

function trainsNORBMLPCriticDCGANCritic()

    # Load labels and images from Flux.Data.sNORB
    @info("Loading data set")
    train_imgs = getsNORBImages()

    batch_size = 32
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:

    generatorInputSize = 20
    @info("Constructing model...")
    wgan = WGAN(DCGANCritic(), MLPGenerator(generatorInputSize = generatorInputSize); generatorInputSize = generatorInputSize, batchSize = batch_size)
   
    trainWGAN(wgan, train_set, train_set; modelName = "sNORB_mlp_dcgan", numSamplesToSave = 40, imageSize = norbImgSize)

end
#getsNORBImages()
    trainsNORBMLP()
#trainsNORBMLPCriticDCGANCritic()

end # module smallNORB
