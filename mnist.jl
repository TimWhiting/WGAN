module mnist
using Flux.Data.MNIST, Statistics
using Base.Iterators: repeated, partition
using Flux

using wgan: WGAN, trainWGAN, DCGANCritic, DCGANGenerator, MLPCritic, MLPGenerator

# Bundle images together with labels and group into minibatchess

function makeMinibatch(X, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    return X_batch
end


function make_minibatch_mlp(X, idxs)
    X_batch = Array{Float32}(undef, 28 * 28, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, i] = reshape(Float32.(X[idxs[i]]), :)
    end
    return X_batch
end


function DCGANCritic()
    model = Chain(x->reshape(x, 28, 28, 1, :),
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
    model = Chain(Dense(generatorInputSize, 288, relu),
        BatchNorm(288),
        x->reshape(x, 3, 3, 32, :),

        # Second convolution, operating upon a 14x14 image
        ConvTranspose((3, 3), 32 => 32, stride = (2, 2), pad = (2, 2)),
        BatchNorm(32, relu),

        # Third convolution, operating upon a 7x7 image
        ConvTranspose((3, 3), 32 => 1, σ, stride = (2, 2), pad = (2, 2)),
        x->reshape(x, 28, 28, :))
    return DCGANGenerator(model)
end

function DCGANCritic2()
    model = Chain(x->reshape(x, 28, 28, 1, :),
        Conv((5, 5), 1 => 32),
        BatchNorm(32, relu),
        x->maxpool(x, (2, 2)),

        # Second convolution, operating upon a 14x14 image
        Conv((5, 5), 32 => 64),
        BatchNorm(64, relu),
        x->maxpool(x, (2, 2)),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x->reshape(x, 1024, :),
        Dense(1024, 1),
    )
    return DCGANCritic(model)
end

function DCGANGenerator2(;generatorInputSize = 100)
    model = Chain(Dense(generatorInputSize, 128 * 7 * 7, relu),
        BatchNorm(128 * 7 * 7),
        x->reshape(x, 7, 7, 128, :),

        # Second convolution, operating upon a 14x14 image
        ConvTranspose((4, 4), 128 => 64, stride = (2, 2), pad = (1, 1)),
        BatchNorm(64, relu),

        # Third convolution, operating upon a 7x7 image
        ConvTranspose((4, 4), 64 => 1, σ, stride = (2, 2), pad = (1, 1)),
        x->reshape(x, 28, 28, :))
    return DCGANGenerator(model)
end



function MLPCritic()
    model = Chain(x->reshape(x, 28^2, :),
        Dense(28^2, 128, relu),
        Dense(128, 1))
    return MLPCritic(model)
end

function MLPGenerator(;generatorInputSize = 10)
    model = Chain(Dense(generatorInputSize, 128, relu),
        Dense(128, 28^2, σ),
        x->reshape(x, 28, 28, :))
    return MLPGenerator(model)
end

function MLPGenerator2(;generatorInputSize = 10)
    model = Chain(Dense(generatorInputSize, 256, relu),
        Dense(256, 28^2, σ),
        x->reshape(x, 28, 28, :))
    return MLPGenerator(model)
end

function trainMNISTMLP()

    # Load labels and images from Flux.Data.MNIST
    @info("Loading data set")
    train_imgs = MNIST.images()

    batch_size = 32
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_set = make_minibatch_mlp(test_imgs, 1:length(test_imgs))

    @info("Constructing model...")
    wgan = WGAN(MLPCritic(), MLPGenerator())
   
    trainWGAN(wgan, train_set, test_set; modelName = "mnist_mlp", numSamplesToSave = 40)

end

function trainMNISTMLPCriticDCGANCritic()

    # Load labels and images from Flux.Data.MNIST
    @info("Loading data set")
    train_imgs = MNIST.images()

    batch_size = 32
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_set = make_minibatch_mlp(test_imgs, 1:length(test_imgs))
    generatorInputSize = 20
    @info("Constructing model...")
    wgan = WGAN(DCGANCritic(), MLPGenerator2(generatorInputSize = generatorInputSize); generatorInputSize = generatorInputSize, batchSize = batch_size)
   
    trainWGAN(wgan, train_set, test_set; modelName = "mnist_mlp_dcgan", numSamplesToSave = 40)

end
function trainMNISTDCGAN()

    # Load labels and images from Flux.Data.MNIST
    @info("Loading data set")
    train_imgs = MNIST.images()

    batch_size = 32
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_set = make_minibatch_mlp(test_imgs, 1:length(test_imgs))
    generatorInputSize = 20
    @info("Constructing model...")
    wgan = WGAN(DCGANCritic2(), DCGANGenerator2(generatorInputSize = generatorInputSize); generatorInputSize = generatorInputSize, batchSize = batch_size)
   
    trainWGAN(wgan, train_set, test_set; modelName = "mnist_mlp_dcgan", numSamplesToSave = 40)

end
#trainMNISTMLP()
trainMNISTMLPCriticDCGANCritic()
#trainMNISTDCGAN()
end # module main
