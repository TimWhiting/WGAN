module wgan
using Juno

using Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, RMSProp
using Base.Iterators: repeated, partition
using Printf, BSON
using learn
using stats
using Statistics

import Flux.Tracker: Params, gradient, data, update!


# Load labels and images from Flux.Data.MNIST
@info("Loading data set")
train_labels = MNIST.labels()
train_imgs = MNIST.images()

batch_size = 128
train_set = makeMinibatches(train_imgs, train_labels, batch_size)

# Prepare test set as one giant minibatch:
test_imgs = MNIST.images(:test)
test_labels = MNIST.labels(:test)
test_set = makeMinibatch(test_imgs, test_labels, 1:length(test_imgs))

abstract type Generator end
abstract type Critic end

struct DCGANGenerator <: Generator
    model
end

struct DCGANCritic <: Critic
    model
end

struct MLPGenerator <: Generator
    model
end

struct MLPCritic <: Critic
    model
end

function MLPCritic()
    model = Chain(Dense(28^2, 128, relu), Dense(128, 32, relu), Dense(32, 1))
    return MLPCritic(model)
end

function MLPGenerator()
    model = Chain(Dense(100, 128, relu), Dense(128, 28^2, σ))
    return MLPGenerator(model)
end

struct WGAN
    α::Float64 # Learning Rate
    c::Float64 # Clipping Parameter
    m::UInt64 # Batch Size
    n_critic::UInt64 # Number of iterations of critic per generator
    critic::Critic # Critic parameters
    generator::Generator # Generator parameters 
end


# NOTES:
# - Make sure to normalize the images, as the generator outputs
# values in (0,1).
# - Try an initial LR of 5e-5. That's what they use in the WGAN paper.
"""
    train!(loss, paramsGenerator, paramsCritic, data, optimizer; cb)

This is an override of Flux.train for a GAN setup

For each datapoint `d` in `data` computes the gradient of `loss(d...)` through
backpropagation and calls the optimizer `opt`.
Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:

```julia
Flux.train!(loss, paramsGenerator, paramsCritic, data, opt,
            cb = throttle(() -> println("training"), 10))
```

The callback can call `Flux.stop()` to interrupt the training loop.

Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(lossGenerator, lossCritic, paramsGenerator, paramsCritic, data, optimizer, postProcessCritic; cb = ()->())
    ps = Params(ps)
    cb = runall(cb)
    @progress for d in data
        try
            for t = 0:model.n_critic;
                # Sample {x^(i)}i=1:m ~ Pr a batch from the real data
                gs = gradient(paramsCritic) do # Make this a batch
                    lossCritic(d...)
                end
                # Sample {z^(i)}i=1:m ~ p(z) a batch of prior samples
                priorgs = gradient(paramsGenerator) do # Make this a batch
                    lossGenerator(d...)
                end
                update!(optimizer, paramsCritic, gs)
                postProcessCritic(paramsCritic) # Do clipping
            end
            priorgs = gradient(paramsGenerator) do # Make this a batch
                lossGenerator(d...)
            end
            update!(optimizer, paramsGenerator, priorgs)
        catch ex
            if ex isa StopException
                break
            else
                rethrow(ex)
            end
        end
    end
end


function clip(params::Params)
    for param in params;
        if param > .1;
            param = .1
        elseif param < -.1;
            param = -.1
        end
    end
end

"""
Calculates the generator's loss, using a sample of Z
"""
generatorLoss(c::Critic, g::Generator, Z) = mean(c(g(Z)))

"""
Calculates the critic's loss, using a sample of `X` and sample
of `Z`, each of equal length. We take the negative of the difference
to turn the output into a loss that we want to minimize.
Minimizing this loss function will maximize the critic's ability
to differentiate between the distribution of the generated data and
the real data.
"""
criticLoss(c::Critic, g::Generator, X, Z) = -(mean(c(X)) - mean(c(g(Z))))

function trainWGAN(wgan::WGAN, trainSet, valSet;
    epochs = 100, targetAcc = 0.999, modelName = "model",
    patience = 10, minLr = 1e-6, lrDropThreshold = 5
)
    modelStats = LearningStats()
    # TODO: Determine what to do for an accuracy function that we can use for the rest of this function
    paramsCritic = Flux.params(wgan.critic.model)
    paramsGenerator = Flux.params(wgan.generator.model)
    opt = RMSProp()

    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:epochs
        global best_acc, last_improvement
        # Train for a single epoch
        train!(generatorLoss, criticLoss, paramsGenerator, paramsCritic, trainSet, opt, clip; cb = wgan.callback)

        # TODO: Figure out how to adapt the rest of this stuff that I got from the model zoo for mnist
        # Calculate accuracy:
        acc = accuracy(valSet...)
        push!(modelStats.valAcc, acc)
        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
        # If our accuracy is good enough, quit out.
        if acc >= targetAcc
            @info(" -> Early-exiting: We reached our target accuracy of $(targetAcc*100)%")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info(" -> New best accuracy! Saving models out to $(modelName)_<type>.bson")
            BSON.@save "$(modelName)_critic.bson" wgan.critic.model epoch_idx acc
            BSON.@save "$(modelName)_generator.bson" wgan.critic.model epoch_idx acc
            best_acc = acc
            modelStats.bestValAcc = best_acc
            last_improvement = epoch_idx
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


#function oldTrainWGAN(model::WGAN, trainingSet::DataSet)
#  while !converged(model.θ)
#      for t = 0:model.n_critic;
#          # Sample {x^(i)}i=1:m ~ Pr a batch from the real data
#          x = sample(trainingSet)
#          # Sample {z^(i)}i=1:m ~ p(z) a batch of prior samples
#          z = sampleGenerator(model.θ)
#          # gw ← ∇w[1/m · sum(fw(x^(i))i=1:m - 1/m · sum(fw(gθ(z^(i))))i=1:m]
#          gw = 0 # Implement this somehow
#          # w ← w + α · RMSProp(w, gw)
#          model.w += model.α*RMSProp(model.w, gw)
#          # w ← clip(w, −c, c)
#          model.w = clip(model.w, -model.c, model.c)
#      end
#      # Sample {z^(i)}i=1:m ∼ p(z) a batch of prior samples
#      z = sampleGenerator(model.θ)
#      # gθ ← −∇θ · 1/m · sum(fw(gθ(z^(i))))i=1:m
#      gθ = 0 # Implement this somehow
#      # θ ← θ − α · RMSProp(θ, gθ)
#      #model.0 -= α * RMSProp(model.θ, gθ)
#  end
#end

end