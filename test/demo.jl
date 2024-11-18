using DISTS
using Images

# Read images
ref = Images.load("../DISTS/images/r0.png")
dist = Images.load("../DISTS/images/r1.png")

# Load the pretrained network parameters and perceptual weights
net_params, weights = load_weights("../DISTS/weights/net_param.mat", "../DISTS/weights/alpha_beta.mat")

# Define resize image flag and use_gpu flag
resize_img = true   # If required. Its usage is not trivial.
use_gpu = false     # GPU acceleration is not implemented

# Calculate the perceptual quality score (DISTS)
@time score = DISTS_score(ref, dist, net_params, weights, resize_img, use_gpu; pretraining = "DISTS")

# Output the score
println("Perceptual quality score: ", score)