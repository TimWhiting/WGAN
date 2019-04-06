# WGAN

Authors: Tim Whiting & Evan Peterson

An implementation of a Wasserstein Generative Adversarial Network (WGAN) in Julia for Advanced Machine Learning class.

## Dependencies

Needed dependencies for running the project locally: `Flux`. If running on GPU with your local machine, also include: `CuArrays`.
Also needs: `Images`, and `ImageMagick` and `NNlib`

## Leveraging TPU

Here is a link to the JuliaTPU repository which tells how to run compile Julia to run on TPU's and gives instructions how to do it on google's colab: https://github.com/JuliaTPU/XLA.jl

## For smallNORB

Get the datasets from here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/
Clone this python repository that converts them all to images: https://github.com/ndrplz/small_norb
Follow the instructions in the repository to convert all of the files to images

## TODOs

- Run for a long time, see if works
- Implement Experiment(s), Options:
  - Bidirectional latent encoder
  - Coordinate convolutions
  - Anything else cool!
  - Validate on a hold-out set from the same distribution as the train set
