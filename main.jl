module main
using Flux.Data.MNIST, Statistics
using Base.Iterators: repeated, partition
push!(LOAD_PATH, "./")
using wgan: WGAN, trainWGAN

#precompile(Flux)
#precompile(wgan)
# Bundle images together with labels and group into minibatchess

function makeMinibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end


function make_minibatch_mlp(X, idxs)
    X_batch = Array{Float32}(undef, 28 * 28, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, i] = reshape(Float32.(X[idxs[i]]), :)
    end
    return X_batch
end

function trainMNIST()

    # Load labels and images from Flux.Data.MNIST
    @info("Loading data set")
    train_imgs = MNIST.images()

    batch_size = 128
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch_mlp(train_imgs, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_set = make_minibatch_mlp(test_imgs, 1:length(test_imgs))

    # Define our model.  We will use a simple convolutional architecture with
    # three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense
    # layer that feeds into a softmax probability output.
    @info("Constructing model...")
    wgan = WGAN()

    # Load model and datasets onto GPU, if enabled
    # train_set = gpu.(train_set)
    # test_set = gpu.(test_set)
    # wgan = gpu(wgan)

    # Make sure our model is nicely precompiled before starting our training loop
    #wgan.critic.model(train_set[1][1])

    trainWGAN(wgan, train_set, test_set)

end

trainMNIST()


end # module main