####################################################
#### EXPERIMENT 1: TRY EITHER KIND OF TRANSPOSE ####
####################################################

# using layers
# using Flux
# using Flux.Tracker: gradient, update!
# using utils
# using LinearAlgebra
# using Random
# using Flux: glorot_uniform

# useFluxVersion = true

# Random.seed!(0)

# shortX = [2 1; 4 4;]
# shortX = reshape(shortX, 2,2,1,1)

# target = [2 2 1 1; 2 2 1 1; 4 4 4 4; 4 4 4 4;]

# myInit = randuFn(0,0.2)

# if useFluxVersion
# using Flux: glorot_uniform
#     myCTLayer = ConvTranspose((3,3), 1=>1, init = glorot_uniform)
# else
#     myCTLayer = ConvolutionTranspose(3, 1, 1, 4, 4, init = glorot_uniform)
# end

# function loss(x, y)
#     ŷ = myCTLayer(x)
#     return sum((y .- ŷ).^2)
# end

# θ = params(myCTLayer)

# if useFluxVersion
#     η = 0.001
#     println(myCTLayer.weight)
# else
#     η = 0.001
#     println(myCTLayer.CTM)
# end

# for i = 1:100
#     g = gradient(() -> loss(shortX, target), θ)
#     for x in θ
#         update!(x, -g[x]*η)
#     end
#     if i % 10 == 0
#         println(loss(shortX, target))
#     end
# end

# println("Final prediction:")
# @show myCTLayer(shortX)
# println("Target:")
# @show target

######################################################################
#### EXPERIMENT 2: USE EITHER CONVOLUTION ON MNIST CLASSIFICATION ####
######################################################################

# Classifies MNIST digits with a convolutional network.
# Writes out saved model to the file "mnist_conv.bson".
# Demonstrates basic model construction, training, saving,
# conditional early-exit, and learning rate scheduling.

# This model, while simple, should hit around 99% test
# accuracy after training for approximately 20 epochs.

using layers
using learn
using stats
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON

EXCLUDE_LABELS = false
LABELS_TO_KEEP = [5,8]
USE_HOMEMADE_CONV = true

# Load labels and images from Flux.Data.MNIST
@info("Loading data set")
train_labels = MNIST.labels()
train_imgs = MNIST.images()

# Pare down the data to just two classes.
if EXCLUDE_LABELS
    two_class_train_indices = train_labels[map(x -> x in LABELS_TO_KEEP, train_labels)]
    train_labels = train_labels[two_class_train_indices]
    train_imgs = train_imgs[two_class_train_indices]
end
@info("Using train set of size $(size(train_imgs))")

# Bundle images together with labels and group into minibatchess
batch_size = 128
train_set = makeMinibatches(train_imgs, train_labels, batch_size)

# Prepare test set as one giant minibatch:
test_imgs = MNIST.images(:test)
test_labels = MNIST.labels(:test)
if EXCLUDE_LABELS
    two_class_test_indices = test_labels[map(x -> x in LABELS_TO_KEEP, test_labels)]
    test_labels = test_labels[two_class_test_indices]
    test_imgs = test_imgs[two_class_test_indices]
end
@info("Using test set of size $(size(test_imgs))")

test_set = makeMinibatch(test_imgs, test_labels, 1:length(test_imgs))

# Define our model.  We will use a simple convolutional architecture with
# three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense
# layer that feeds into a softmax probability output.
@info("Constructing model...")
if USE_HOMEMADE_CONV
    model = Chain(
        Convolution(5, 1, 16, relu),
        Convolution(5, 16, 32, relu),
        Convolution(5, 32, 10, relu),
        x -> reshape(x, :, size(x, 4)),
        # x -> println("new size == $(size(x))"),
        Dense(2560, 10),
        softmax
    )
else
    model = Chain(
        Conv((5, 5), 1=>16, relu),
        Conv((5, 5), 16=>32, relu),
        Conv((5, 5), 32=>10, relu),
        x -> reshape(x, :, size(x, 4)),
        # x -> println("new size == $(size(x))"),
        Dense(2560, 10),
        softmax,
    )
end

# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set)
test_set = gpu.(test_set)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust.
function loss(x, y)
    # We augment `x` a little bit here, adding in random noise
    x_aug = x .+ 0.1*gpu(randn(eltype(x), size(x)))
    y_hat = model(x_aug)
    return crossentropy(y_hat, y)
end

# Train our model with the given training set using the ADAM optimizer and
# printing out performance against the test set as we go.
opt = ADAM(0.001)
evalcb = () -> @show(loss(train_set[1]...))

results = trainOne(loss, model, train_set, test_set, modelName = "mnist_conv", epochs = 2)

plotLearningStats(results, "mnist_conv_results", true)

# Note - Flux version of MNIST was able to get validation set accuracy of 0.9781
# after just 2 epochs.