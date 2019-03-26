# WGAN

Authors: Tim Whiting & Evan Peterson

An implementation of a Wasserstein Generative Adversarial Network (WGAN) in Julia for Advanced Machine Learning class.

## Dependencies

Needed dependencies for running the project locally: `Flux`. If running on GPU with your local machine, also include: `CuArrays`.

## Leveraging TPU

Here is a link to the JuliaTPU repository which tells how to run compile Julia to run on TPU's and gives instructions how to do it on google's colab: https://github.com/JuliaTPU/XLA.jl

## TODOs

- Implement gradient clipping
- (Tim) Implement Earth Mover's Distance metric
- Implement generator net
- (Tim) Implement critic net
- Implement Experiment(s), Options:
  - Bidirectional latent encoder
  - Coordinate convolutions
  - Anything else cool!
  - Validate on a hold-out set from the same distribution as the train set