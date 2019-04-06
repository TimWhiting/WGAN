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

# useFluxVersion = false

# Random.seed!(0)

# shortX = [2 1; 4 4;]
# shortX = reshape(shortX, 2,2,1,1)

# target = [2 2 1 1; 2 2 1 1; 4 4 4 4; 4 4 4 4;]

# myInit = randuFn(0,0.2)

# if useFluxVersion
#     myCTLayer = ConvTranspose((3,3), 1=>1)
# else
#     myCTLayer = ConvolutionTranspose(3, 1, 1, 4, 4)
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

using Flux, Flux.Data.MNIST, Statistics, Printf, BSON
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition

using layers, learn, stats

function main(; useConv = true, runHomemade = true, runFlux = true, modelName = "mnist_model", epochs = 5)

    if useConv
        test_set = makeTwoClassShapes((5,5))
        train_set = [test_set]
    else
        # Load labels and images from Flux.Data.MNIST
        @info("Loading data set")
        train_labels = MNIST.labels()
        train_imgs = MNIST.images()
        @info("Using train set of size $(size(train_imgs))")

        # Bundle images together with labels and group into minibatchess
        batch_size = 128
        train_set = makeMinibatches(train_imgs, train_labels, batch_size)

        # Prepare test set as one giant minibatch:
        test_imgs = MNIST.images(:test)
        test_labels = MNIST.labels(:test)
        @info("Using test set of size $(size(test_imgs))")

        test_set = makeMinibatch(test_imgs, test_labels, 1:length(test_imgs))
    end

    # Define our model.  We will use a simple convolutional architecture with
    # three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense
    # layer that feeds into a softmax probability output.
    @info("Constructing model...")
    if useConv
        homemadeModel = Chain(
            Convolution(3, 1, 2, σ),
            Convolution(3, 2, 2, σ),
            x -> reshape(x, :, size(x, 4)),
            # x -> println("new size == $(size(x))"),
            softmax
        )
        model = Chain(
            Conv((3, 3), 1=>2, σ),
            Conv((3, 3), 2=>2, σ),
            x -> reshape(x, :, size(x, 4)),
            # x -> println("new size == $(size(x))"),
            softmax,
        )
    else
        homemadeModel = Chain(
            x -> reshape(x, 28^2, :),
            Connected(28^2, 32, relu),
            Connected(32, 10),
            softmax
        )
        model = Chain(
            x -> reshape(x, 28^2, :),
            Dense(28^2, 32, relu),
            Dense(32, 10),
            softmax
        )
    end

    # Load model and datasets onto GPU, if enabled
    # train_set = gpu.(train_set)
    # test_set = gpu.(test_set)
    # model = gpu(model)

    @info("Precompiling models...")
    # Make sure our model is nicely precompiled before starting our training loop
    model(train_set[1][1])
    homemadeModel(train_set[1][1])

    # `loss()` calculates the crossentropy loss between our prediction `y_hat`
    # (calculated from `model(x)`) and the ground truth `y`.  We augment the data
    # a bit, adding gaussian random noise to our image to make it more robust.
    function convLoss(m, x, y)
        # We augment `x` a little bit here, adding in random noise
        x_aug = x .+ 0.1*gpu(randn(eltype(x), size(x)))
        y_hat = m(x_aug)
        return crossentropy(y_hat, y)
    end

    denseLoss(m, x, y) = crossentropy(m(x), y)

    allResults::Array{LearningStats} = []
    modelNames::Array{String} = []

    if runFlux
        results = trainOne(
            useConv ? convLoss : denseLoss, model, train_set, test_set, ADAM(0.01),
            modelName = "$(modelName)_results", epochs = epochs, save = false,
            patience = 1000, lrDropThreshold = 1000, reportEvery = 10, earlyExit = false
        )
        # plotLearningStats(results, "$(modelName)_results", true)
        push!(allResults, results)
        push!(modelNames, "Flux")
    end

    if runHomemade
        homemadeResults = trainOne(
            useConv ? convLoss : denseLoss, homemadeModel, train_set, test_set, ADAM(0.01),
            modelName = "$(modelName)_homemade_results", epochs = epochs, save = false,
            patience = 1000, lrDropThreshold = 1000, reportEvery = 10, earlyExit = false
        )
        # plotLearningStats(homemadeResults, "$(modelName)_homemade_results", true)
        push!(allResults, homemadeResults)
        push!(modelNames, "Our")
    end

    plotCompareModels(allResults, modelNames, "$(modelName)_comparison")
end

main(useConv = true, runFlux = true, runHomemade = true, epochs = 500, modelName = "boxsquare_conv")

# Note - Flux version of MNIST was able to get validation set accuracy of 0.9803
# after just 2 epochs.

#########################################################
#### EXPERIMENT 3: TESTING THE HOMEMADE CONVOLUTIONS ####
#########################################################

# using layers

# # Source: https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0

# wInit(dims...) = reshape([1 4 1; 1 4 3; 3 3 1;], 3, 3, 1, 1)

# # Testing the normal convolution

# x = reshape(transpose([4 5 8 7; 1 8 8 8; 3 6 6 4; 6 5 7 8;]), 4, 4, 1, 1)
# expectedOut = transpose([122 148; 126 134;])

# myConv = Convolution(3, 1, 1, 4, 4, init = wInit)
# out = myConv(x)
# @info("For the Convolution:")
# println("Actual output:\n$(out)")
# println("Expected output:\n$(expectedOut)")

# # Testing the convolution transpose

# xT = reshape(transpose([2 1; 4 4;]), 2, 2, 1, 1)
# expectedOutT = transpose([2 9 6 1; 6 29 30 7; 10 29 33 13; 12 24 16 4;])

# myConvT = ConvolutionTranspose(3, 1, 1, 4, 4, init = wInit)
# outT = myConvT(xT)
# @info("For the Convolution Transpose:")
# println("Actual output:\n$(outT)")
# println("Expected output:\n$(expectedOutT)")

###########################################################################
#### EXPERIMENT 4: USE EITHER CONVOLUTION ON BOX/SQUARE CLASSIFICATION ####
###########################################################################

# using layers
# using learn
# using stats
# using datasets
# using Flux, Statistics
# using Flux: onehotbatch, onecold, crossentropy, mse, throttle
# using Base.Iterators: repeated, partition
# using Printf, BSON

# function main(; useHomemadeConv = false)

#     @info("Loading data set")
#     train_set = makeTwoClassShapes((5,5))
#     test_set = train_set
#     println(train_set)
#     train_set = [train_set]

#     # Define our model. We will use a simple convolutional architecture with
#     # a final Dense layer that feeds into a softmax probability output.
#     @info("Constructing model...")
#     if useHomemadeConv
#         model = Chain(
#             Convolution(5, 1, 1, σ),
#             # Convolution(3, 2, 2, σ),
#             x -> reshape(x, :, size(x, 4)),            
#             # x -> println("new size == $(size(x))"),
#             # Dense(2, 2),
#             # softmax
#         )
#     else
#         model = Chain(
#             Conv((5, 5), 1=>1, σ),
#             # Conv((3, 3), 2=>2, σ),
#             x -> reshape(x, :, size(x, 4)),            
#             # x -> println("new size == $(size(x))"),
#             # Dense(2, 2),
#             # softmax,
#         )
#     end

#     @info("Precompiling model...")
#     # Make sure our model is nicely precompiled before starting our training loop
#     model(train_set[1][1])

#     if useHomemadeConv
#         modelName = "boxsquare_homemade_conv"
#     else
#         modelName = "boxsquare_conv"
#     end

#     function round(val::Flux.Tracker.TrackedReal{Float32})
#         if val >= .5
#             return Flux.Tracker.TrackedReal{Float32}(1.)
#         else
#             return Flux.Tracker.TrackedReal{Float32}(0.)
#         end
#     end 

#     # `loss()` calculates the crossentropy loss between our prediction `y_hat`
#     # (calculated from `model(x)`) and the ground truth `y`.  We augment the data
#     # a bit, adding gaussian random noise to our image to make it more robust.
#     function loss(x, y)
#         ŷ = model(x)
#         return mean((y .- ŷ).^2)
#     end

#     function accuracy(model, x, y)
#         ŷ = model(x)
#         return mean(round.(ŷ) .== y) 
#     end

#     results = trainOne(
#         loss, model, train_set, test_set, Momentum(0.2), 
#         modelName = modelName, epochs = 1000, save = false,
#         lrDropThreshold = 100, earlyExit = false, reportEvery = 10,
#         cb = () -> (), patience = 1000, showWeights = false,
#         accuracy = accuracy
#     )

#     plotLearningStats(results, "$(modelName)_results", true)

# end

# main(useHomemadeConv = true)

########################################################
#### EXPERIMENT 5: TESTING LEARNING OF CONVOLUTIONS ####
########################################################

# using Flux
# using Flux.Tracker: gradient, update!
# using Flux: glorot_uniform
# using LinearAlgebra, Random

# using utils, layers, datasets

# function main(; useFluxVersion = false)

#     Random.seed!(0)

#     data, labels = makeTwoClassShapes((5,5))
#     n = size(labels)[1]
#     data = [reshape(data[:, :, :, i], 5, 5, 1, 1) for i in 1:n]
#     labels = [labels[i] for i in 1:n]
#     @show data
#     @show labels

#     # data = [
#     #     [0 0 0 0 0; 0 1 1 1 0; 0 1 1 1 0; 0 1 1 1 0; 0 0 0 0 0;],
#     #     [0 0 0 0 0; 0 1 1 1 0; 0 1 0 1 0; 0 1 1 1 0; 0 0 0 0 0;]
#     # ]
#     # x = map(x -> reshape(x, 5,5,1,1), x)

#     # labels = [0, 1]

#     if useFluxVersion
#         model = Chain(Conv((5,5), 1=>1))
#     else
#         model = Chain(Convolution(5, 1, 1))
#     end

#     function loss(x, y)
#         ŷ = model(x)
#         return mean((y .- ŷ).^2)
#     end

#     θ = params(model)
#     @show θ

#     if useFluxVersion
#         @show model[1].weight
#     else
#         @show model[1].W    
#     end

#     η = 0.01
#     for i = 1:1000
#         for j in 1:2
#             g = gradient(() -> loss(data[j], labels[j]), θ)
#             for x in θ
#                 update!(x, -g[x]*η)labels
#             end
#             if i % 100 == 0 && j == 1
#                 println(loss(data[j], data[j]))
#             end
#         end
#     end

#     @show θ
#     println("Final prediction:")
#     @show model(data[1])
#     println("Target:")
#     @show labels[1]

# end

# main(useFluxVersion = false)