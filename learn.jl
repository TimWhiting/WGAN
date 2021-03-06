module learn

using layers
using stats
using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: partition
using Printf, BSON

# Bundle images together with labels and group into minibatchess
function makeMinibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

function makeMinibatches(X, Y, batchSize)
    mbIdxs = partition(1:length(X), batchSize)
    minibatches = [makeMinibatch(X, Y, i) for i in mbIdxs]
    return minibatches
end

oneHotClassificationAccuracy(model, x, y) = mean(onecold(model(x)) .== onecold(y))

"""
Train a single model with the given training set and sensible defaults.
Prints out performance against the validation set as we go.
Note that `trainSet` should be multiple batches, and `valSet` should be
just one batch.
"""
function trainOne(loss, model, trainSet, valSet, opt = ADAM(0.001);
    cb = () -> (), epochs = 100,
    targetAcc = 0.999, modelName = "model",
    patience = 10, minLr = 1e-6, lrDropThreshold = 5,
    save = true, earlyExit = true, reportEvery = 1,
    showWeights = false, accuracy = oneHotClassificationAccuracy
)
    @info("Beginning training loop...")
    best_acc = 0.0
    best_loss = Inf
    last_improvement = 0
    modelStats = LearningStats()
    lossWithModel(x, y) = loss(model, x, y)

    for epoch_idx in 1:epochs
        best_acc, last_improvement
        # Train for a single epoch
        Flux.train!(lossWithModel, params(model), trainSet, opt, cb = throttle(cb, 10))

        # Calculate accuracy:
        acc = accuracy(model, valSet...)
        push!(modelStats.valAcc, acc)
        if epoch_idx % reportEvery == 0
            @info(@sprintf("[%d]: Validation accuracy: %.4f", epoch_idx, acc))
        end

        # Calculate training loss:
        trainLoss = lossWithModel(trainSet[1]...)
        push!(modelStats.trainLoss, trainLoss)        

        # Calculate validation loss:
        valLoss = lossWithModel(valSet...)
        push!(modelStats.valLoss, valLoss)
        if valLoss < best_loss
            best_loss = valLoss
            modelStats.bestValLoss = best_loss
            last_improvement = epoch_idx
        end
        if epoch_idx % reportEvery == 0
            @info(@sprintf("[%d]: Validation loss: %.6f", epoch_idx, valLoss))
        end
        
        # If our accuracy is good enough, quit out.
        if earlyExit && acc >= targetAcc
            @info(" -> Early-exiting: We reached our target accuracy of $(targetAcc*100)%")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            if epoch_idx % reportEvery == 0
                @info(" -> New best accuracy!")
            end
            if save
                @info(" -> Saving model out to $(modelName).bson")
                BSON.@save "$(modelName).bson" model epoch_idx acc
            end
            best_acc = acc
            modelStats.bestValAcc = best_acc
        end

        if showWeights
            @info("Model Weights:")
            for (i, layer) in enumerate(model)
                if :W in fieldnames(typeof(layer))
                    @info(@sprintf("[%d]th layer weights:", i))
                    @info(layer.W)
                end
                if :weight in fieldnames(typeof(layer))
                    @info(@sprintf("[%d]th layer weights:", i))
                    @info(layer.weight)
                end
                if :b in fieldnames(typeof(layer))
                    @info(@sprintf("[%d]th layer bias weights:", i))
                    @info(layer.b)
                end
            end
        end

        # If we haven't seen improvement in lrDropThreshold epochs, drop our learning rate:
        if epoch_idx - last_improvement >= lrDropThreshold && opt.eta > minLr
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= patience
            @warn(" -> We're calling this converged.")
            break
        end
    end
    return modelStats
end

export makeMinibatch, makeMinibatches, trainOne, oneHotClassificationAccuracy

end # module learn