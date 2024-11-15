# DISTS.jl
[![Build Status](https://github.com/gsahonero/DISTS.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/gsahonero/DISTS.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia implementation of Deep Image Structure and Texture Similarity (DISTS) metric. This implementation follows the logic implemented in [DISTS](https://github.com/dingkeyan93/DISTS/tree/master). Please, check [comments](#comments) before using.

## Setup
For now: 

```julia
using Pkg;
Pkg.add(url="https://github.com/gsahonero/DISTS.jl")
Pkg.dev("DISTS")
```

## Usage

The classic example: 

```julia
using DISTS
using Images

# Read images
ref = Images.load("../images/r0.png")
dist = Images.load("../images/r1.png")

# Load the pretrained network parameters and perceptual weights
net_params, weights = load_weights("../DISTS/weights/net_param.mat", "../DISTS/weights/alpha_beta.mat")

# Define resize image flag and use_gpu flag
resize_img = false   # If required. Its usage is not trivial.
use_gpu = false     # GPU acceleration is not implemented yet

# Calculate the perceptual quality score (DISTS)
score = DISTS_score(ref, dist, net_params, weights, resize_img, use_gpu; pretraining = "DISTS")

# Output the score
println("Perceptual quality score: ", score)
```

## Parameters of `DISTS_score`
- `ref`:`Image` - reference image
- `dist`:`Image` - distorted image
- `net_params`: `Dict` - holds the weights from `net_params.mat` available at the original repository.
- `weights`: `Dict` - holds the alpha-beta weights from `alpha_beta.mat`.
- `resize_img`: `Boolean` - to enable the use of `imresize` when applies
- `use_gpu`: `Boolean` - to enable GPU acceleration (not implemented yet)
- `pretraining`:`String` - defines which pretraining setup will be used. By default set to "DISTS". It supports "Flux" an "Mixed". "Flux" loads the 16 layers of VGG without modification. "Mixed" loads the convolutional layers of VGG without modification, but the pooling layers weights use the weights from `net_params.mat`

## Comments
- (AFAIK) Julia does not have an implementation of the DISTS metric, this "package" should do the trick. However, Flux implementation seems to produce different results with respect to PyTorch and MATLAB despite having the same weights.
- Pooling layers from DISTS are implemented as DepthwiseConvolutions.