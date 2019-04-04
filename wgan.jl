module wgan
using Juno
using Flux.Data.MNIST, Statistics
using Flux
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

const ϵ = 1e-8

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

function randu(dims::Tuple{Vararg{Int64}})
    return Float32.(rand(dims...) .* 2 .- 1.)
end
function randGaussian(dims::Tuple{Vararg{Int64}}, mean::Float32, stddev::Float32)::Array{Float32}
    return Float32.((randn(dims) .* stddev) .- (stddev / 2) .+ mean)
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

function DCGANGenerator()
    model = Chain(Dense(100, 288),
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
    model = Chain(x->reshape(x, 28^2, :), Dense(28^2, 128, σ), Dense(128, 1))
    return MLPCritic(model)
end

function MLPGenerator()
    model = Chain(Dense(10, 128, σ), Dense(128, 28^2, σ), x->reshape(x, 28, 28, :))
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

# TODO: Make default parameters good
function WGAN(;learningRate = Float32(.0001),clippingParam =  Float32(.01), batchSize = 32, generatorInputSize = 10, nCriticIterationsPerGeneratorIteration = 5, dcganCritic = false, dcganGenerator = false)
    if dcganCritic
        if dcganGenerator
            return WGAN(learningRate, clippingParam, batchSize, generatorInputSize, nCriticIterationsPerGeneratorIteration, DCGANCritic(), DCGANGenerator(), ()->())
        else
            return WGAN(learningRate, clippingParam, batchSize, generatorInputSize, nCriticIterationsPerGeneratorIteration, DCGANCritic(), MLPGenerator(), ()->())
        end
    else
        return WGAN(learningRate, clippingParam, batchSize, generatorInputSize, nCriticIterationsPerGeneratorIteration, MLPCritic(), MLPGenerator(), ()->())
    end
end


# NOTES:
# - Make sure to normalize the images, as the generator outputs
# values in (0,1).
# - Try an initial LR of 5e-5. That's what they use in the WGAN paper.

struct StopException <: Exception end
call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = ()->foreach(call, fs)
data(x::TrackedArray) = x.data
tracker(x::TrackedArray) = x.tracker

function update_clip!(x::TrackedArray, Δ; c = .01)
    x.data .+= data(Δ)
    for (index, xd) in enumerate(x.data)
        if xd > c
            x.data[index] = c
        elseif xd < -c
            x.data[index] = -c
        end
    end
    tracker(x).grad .= 0
    return x
end

function apply!(o::RMSProp, x, Δ)
    η, ρ = o.eta, o.rho
    acc = get!(o.acc, x, zero(x))::typeof(data(x))
    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. Δ *= η / (√acc + ϵ)
end
  

function update_pos!(opt, x, x̄)
    update_clip!(x, apply!(opt, x, data(x̄)))
end
  
  function update_pos!(opt, xs::Params, gs)
    for x in xs
        update_pos!(opt, x, gs[x])
    end
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
function train!(lossGenerator, lossCritic, wgan::WGAN, data, optGenerator, optCritic; cb = ()->())
    paramsCritic = Params(params(wgan.critic.model))
    paramsGenerator = Params(params(wgan.generator.model))
    cb = runall(cb)
    t = 1
    @progress for d in data
        try
            if t % wgan.n_critic == 0 # If this is the nth batch, do both critic and generator update               
                gs = gradient(paramsCritic) do # Make this a batch
                    lossCritic(wgan.critic, wgan.generator, d, randu((wgan.n, wgan.m)))
                end
                update_pos!(optCritic, paramsCritic, gs)
                #postProcessCritic(paramsCritic, wgan.c)
                priorgs = gradient(paramsGenerator) do # Make this a batch
                    lossGenerator(wgan.critic, wgan.generator, randu((wgan.n, wgan.m)))
                end
                update!(optGenerator, paramsGenerator, priorgs)
            else
                # Sample {x^(i)}i=1:m ~ Pr a batch from the real data
                # Sample {z^(i)}i=1:m ~ p(z) a batch of prior samples
                gs = gradient(paramsCritic) do # Make this a batch
                    lossCritic(wgan.critic, wgan.generator, d, randu((wgan.n, wgan.m)))
                end
                update_pos!(optCritic, paramsCritic, gs)
                #postProcessCritic(paramsCritic, wgan.c)
            end
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


"""
Calculates the generator's loss, using a sample of Z
"""
generatorLoss(c::Critic, g::Generator, Z::AbstractArray{Float32,2}) = -mean(c.model(g.model(Z)))

"""
Calculates the critic's loss, using a sample of `X` and sample
of `Z`, each of equal length. We take the negative of the difference
to turn the output into a loss that we want to minimize.
Minimizing this loss function will maximize the critic's ability
to differentiate between the distribution of the generated data and
the real data.
"""
criticLoss(c::Critic, g::Generator, X::AbstractArray, Z::AbstractArray{Float32,2}) = (mean(c.model(X)) - mean(c.model(g.model(Z))))

function trainWGAN(wgan::WGAN, trainSet, valSet;
    epochs = 100, targetLoss = -500, modelName = "model",
    patience = 10, minLr = 1e-6, lrDropThreshold = 5)
    @info("Beginning training function...")
    modelStats = LearningStats()
    optCritic = RMSProp(wgan.α)
    optGenerator = RMSProp(wgan.α)
    mkpath("images/mnist_mlp/")
    save("images/mnist_mlp/image_epoch_sample.png", colorview(Gray, reshape(trainSet[1], 28, 28, 1, :)[:,:,:,1]))
    @info("Beginning training loop...")
    best_loss = 10000000000000000000000000000.0
    last_improvement = 0
    for epoch_idx in 1:epochs
        # Train for a single epoch
        
        #gpu(wgan.critic.model)
        #gpu(wgan.generator.model)
        #gpu.(trainSet)

        train!(generatorLoss, criticLoss, wgan, trainSet, optGenerator, optCritic; cb = wgan.callback)
    
        # Calculate loss:
        loss = criticLoss(wgan.critic, wgan.generator, trainSet[1], randu((wgan.n, wgan.m)))
        push!(modelStats.valAcc, loss)
        @info(@sprintf("[%d]: Test loss: %.4f", epoch_idx, loss))
     
        save("images/mnist_mlp/image_epoch_$(epoch_idx).png", colorview(Gray, reshape(wgan.generator.model(randu((wgan.n, 1))), 28, 28)))
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
        #if epoch_idx - last_improvement >= lrDropThreshold && opt.eta > minLr
            #opt.eta /= 10.0
            #@warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
            # After dropping learning rate, give it a few epochs to improve
            #last_improvement = epoch_idx
        #end

        #if epoch_idx - last_improvement >= patience
        #    @warn(" -> We're calling this converged.")
        #    break
        #end
    end
    return modelStats
end

end