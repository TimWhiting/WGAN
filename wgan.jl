module wgan
using Juno

using Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, RMSProp
using Base.Iterators: repeated, partition
using Printf, BSON

import Flux.Tracker: Params, gradient, data, update!


# Load labels and images from Flux.Data.MNIST
@info("Loading data set")
train_labels = MNIST.labels()
train_imgs = MNIST.images()
# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end
batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

# Prepare test set as one giant minibatch:
test_imgs = MNIST.images(:test)
test_labels = MNIST.labels(:test)
test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

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
    model = Chain(Dense(28, 28), Dense(28, 14), Dense(14, 10), softmax)
    return MLPCritic(model)
end

function MLPGenerator()
    model = Chain(Dense(10, 14), Dense(14, 28), Dense(28, 28))
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
        if param > .1
            param = .1
        else if param < -.1
                param = -.1
            end
        end
    end
end


"""
  earthMoversDistance(x, y)
Calculates the earth movers distance loss function
   `x` is the given data sample
   `y` is the expected distribution   
"""
function EarthMoversDistance(x, y)
# TODO: create earthMoversDistance loss function
end

function trainWGAN(wgan::WGAN, trainingSet::DataSet)
    lossGenerator(x, y) = EarthMoversDistance(wgan.generator.model(x), y);
    # TODO: What is the loss of critic
    lossCritic(x, y) = EarthMoversDistance(wgan.critic.model(x), y);
    paramsCritic = Flux.params(wgan.critic.model)
    paramsGenerator = Flux.params(wgan.generator.model)
    train!(lossGenerator, lossCritic, paramsGenerator, paramsCritic, trainingSet, RMSProp, clip; cb = wgan.callback)
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