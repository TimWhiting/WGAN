module wgan
using Juno

using Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, RMSProp, Dense, Chain, params, Params, mapparams
using Base.Iterators: repeated, partition
using Printf, BSON
using learn
using stats
using Statistics
using NNlib: relu, σ
import Flux.Tracker: Params, gradient, data, update!


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
    α::Float32 # Learning Rate
    c::Float32 # Clipping Parameter
    m::Int64 # Batch Size
    n::Int64 # Input to generator size
    n_critic::UInt64 # Number of iterations of critic per generator
    critic::Critic # Critic parameters
    generator::Generator # Generator parameters 
    callback::Function
end

function WGAN()
    # TODO: Add versions with DCGAN, remember batch normalization
    # TODO: Fix these parameters
    return WGAN(Float32(.001), Float32(.4), 10, 100, 10, MLPCritic(), MLPGenerator(), ()->())
end

function randGaussian(dims::Tuple{Vararg{Int64}}, mean::Float32, stddev::Float32)::Array{Float32}
    return Float32.((randn(dims) .* stddev) .- (stddev / 2) .+ mean)
end
# NOTES:
# - Make sure to normalize the images, as the generator outputs
# values in (0,1).
# - Try an initial LR of 5e-5. That's what they use in the WGAN paper.

struct StopException <: Exception end
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = ()->foreach(call, fs)
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
function train!(lossGenerator, lossCritic, wgan::WGAN, data, optimizer, postProcessCritic; cb = ()->())
    paramsCritic = Params(params(wgan.critic.model))
    paramsGenerator = Params(params(wgan.generator.model))
    cb = runall(cb)
    
    @progress for d in data
        try
            for t = 0:wgan.n_critic;
                # Sample {x^(i)}i=1:m ~ Pr a batch from the real data
                # Sample {z^(i)}i=1:m ~ p(z) a batch of prior samples
                gs = gradient(paramsCritic) do # Make this a batch
                    lossCritic(wgan.critic, wgan.generator, d, randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0)))
                end
                update!(optimizer, paramsCritic, gs)
                postProcessCritic(paramsCritic, wgan.c)
            end
            priorgs = gradient(paramsGenerator) do # Make this a batch
                lossGenerator(wgan.critic, wgan.generator, randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0)))
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


function clip(params::Params, c::Float32)
    mapparams(params) do param
        if param > c
            param = c
        elseif param < -c
            param = -c
        end
    end
end

"""
Calculates the generator's loss, using a sample of Z
"""
generatorLoss(c::Critic, g::Generator, Z::AbstractArray{Float32,2}) = mean(c.model(g.model(Z)))

"""
Calculates the critic's loss, using a sample of `X` and sample
of `Z`, each of equal length. We take the negative of the difference
to turn the output into a loss that we want to minimize.
Minimizing this loss function will maximize the critic's ability
to differentiate between the distribution of the generated data and
the real data.
"""
criticLoss(c::Critic, g::Generator, X::AbstractArray{Float32,2}, Z::AbstractArray{Float32,2}) = -(mean(c.model(X)) - mean(c.model(g.model(Z))))

function trainWGAN(wgan::WGAN, trainSet, valSet;
    epochs = 100, targetLoss = 0.001, modelName = "model",
    patience = 10, minLr = 1e-6, lrDropThreshold = 5)
    @info("Beginning training function...")
    modelStats = LearningStats()
    opt = RMSProp()

    @info("Beginning training loop...")
    best_loss = 0.0
    last_improvement = 0
    for epoch_idx in 1:epochs
        # Train for a single epoch
        train!(generatorLoss, criticLoss, wgan, trainSet, opt, clip; cb = wgan.callback)

        # Calculate loss:
        loss = -criticLoss(wgan.critic, wgan.generator, trainSet[1], randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0)))
        push!(modelStats.valAcc, loss)
        @info(@sprintf("[%d]: Test loss: %.4f", epoch_idx, loss))
    
        # If our loss is good enough, quit out.
        if targetLoss >= loss
            @info(" -> Early-exiting: We reached our target loss of $(targetLoss)")
            break
        end

        # If this is the best loss we've seen so far, save the model out
        if best_loss >= loss
            @info(" -> New best loss! Saving models out to $(modelName)_<type>.bson")
            #TODO: FIgure out saving models
            #BSON.@save "$(modelName)_critic.bson" wgan.critic.model epoch_idx loss
            #BSON.@save "$(modelName)_generator.bson" wgan.critic.model epoch_idx loss
            best_loss = loss
            modelStats.bestValAcc = best_loss
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