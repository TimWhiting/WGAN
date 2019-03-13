# WGAN
An implementation of a WGAN in Julia for Advanced Machine Learning class.

## Dependencies

Needed dependencies for running the project locally: `Flux`. If running on GPU with your local machine, also include: `CuArrays`.

## Leveraging TPU

Here is a link to the JuliaTPU repository which tells how to run compile Julia to run on TPU's and gives instructions how to do it on google's colab: https://github.com/JuliaTPU/XLA.jl

## TODOs

- Implement fully connected layer
- Implement deconvolution layer
- Implement gradient clipping
- Implement Earth Mover's Distance metric
- Implement generator net
- Implement critic net