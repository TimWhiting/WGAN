module wgan
using Juno

using Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, RMSProp, Dense, Chain, params, Params, mapparams
using Base.Iterators: repeated, partition
using Printf
using BSON: @save
using learn
using stats
using Statistics
using Images
using Dates: now
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
    model = Chain(
        Dense(28^2, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 1))
    return MLPCritic(model)
end

function MLPGenerator()
    model = Chain(
        Dense(100, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 28^2, σ)
    )
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
    return WGAN(Float32(.00005), Float32(.01), 64, 100, 5, MLPCritic(), MLPGenerator(), ()->())
end

function randGaussian(dims::Tuple{Vararg{Int64}}, mean::Float32, stddev::Float32)::Array{Float32}
    return Float32.((randn(dims) .* stddev) .- (stddev / 2) .+ mean)
end

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
    t = 0
    @progress for d in data
        try
            if t % wgan.n_critic == 0 # If this is the nth batch, do both critic and generator update               
                gs = gradient(paramsCritic) do # Make this a batch
                    lossCritic(wgan.critic, wgan.generator, d, randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0)))
                end
                update!(optimizer, paramsCritic, gs)
                postProcessCritic(paramsCritic, wgan.c)
                priorgs = gradient(paramsGenerator) do # Make this a batch
                    lossGenerator(wgan.critic, wgan.generator, randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0)))
                end
                update!(optimizer, paramsGenerator, priorgs)
            else
                # Sample {x^(i)}i=1:m ~ Pr a batch from the real data
                # Sample {z^(i)}i=1:m ~ p(z) a batch of prior samples
                gs = gradient(paramsCritic) do # Make this a batch
                    lossCritic(wgan.critic, wgan.generator, d, randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0)))
                end
                update!(optimizer, paramsCritic, gs)
                postProcessCritic(paramsCritic, wgan.c)
            end
            cb()
            t += 1
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
    patience = 10, minLr = 1e-6, lrDropThreshold = 5
)
    modelStats = LearningStats()
    opt = RMSProp()
    function lossCb()
        @show(criticLoss(wgan.critic, wgan.generator, trainSet[1], randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0))))
        @show(generatorLoss(wgan.critic, wgan.generator, randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0))))
    end

    @info("Beginning training loop...")
    best_loss = Inf32
    last_improvement = 0
    for epoch_idx in 1:epochs
        # Train for a single epoch
        train!(generatorLoss, criticLoss, wgan, trainSet, opt, clip; cb = throttle(lossCb, 10))

        # Calculate loss:
        loss = criticLoss(wgan.critic, wgan.generator, trainSet[1], randGaussian((wgan.n, wgan.m), Float32(0.0), Float32(1.0)))
        push!(modelStats.valAcc, loss)
        @info(@sprintf("[%d]: Test loss: %.4f", epoch_idx, loss))
     
        save("images/$(modelName)/image_epoch_$(epoch_idx).png", colorview(Gray, reshape(wgan.generator.model(randGaussian((wgan.n, 1), Float32(0.0), Float32(1.0))), 28, 28)))
        # If our loss is good enough, quit out.
        if targetLoss >= loss
            @info(" -> Early-exiting: We reached our target loss of $(targetLoss)")
            break
        end

        # If this is the best loss we've seen so far, save the model out
        if best_loss >= loss
            @info(" -> New best loss! Saving models out to $(modelName)_critic/generator-timestamp.bson")
            #TODO: Figure out saving models -- not working right now
            #critic = wgan.critic.model
            #generator = wgan.generator.model
            #@save "$(modelName)_critic-$(now()).bson" critic epoch_idx loss
            #@save "$(modelName)_generator-$(now()).bson" generator epoch_idx loss
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

end