using Flux
using Metalhead
using CUDA
using Images
using ImageTransformations
using Statistics
using MAT

# It organizes to have the same order of Metalhead vgg weights
function fix_weights_order(net_params, layer_name)
    layer_size = size(net_params[layer_name])
    result = zeros(layer_size)
    for i=1:layer_size[3]
        for j=1:layer_size[4]
            result[:,:,i,j] = reverse(net_params[layer_name][:,:,i,j]', dims=(1,2))
        end
    end
    return result
end

# Function to load the pretrained weights (net_params and alpha_beta)
function load_weights(net_param_file, alpha_beta_file)
    # Load the net_params (VGG16 parameters) from a .mat file
    net_params = matread(net_param_file)
    fixed_net_params = deepcopy(net_params)
    # Flux weights are handled differently, when loading weights this way we bridge the gap between those computations. These implementations could be faster and shorter. Biases do not require a fix.
    fixed_net_params["conv1_1_weight"] = fix_weights_order(net_params, "conv1_1_weight")
    fixed_net_params["conv1_2_weight"] = fix_weights_order(net_params, "conv1_2_weight")
    fixed_net_params["conv2_1_weight"] = fix_weights_order(net_params, "conv2_1_weight")
    fixed_net_params["conv2_2_weight"] = fix_weights_order(net_params, "conv2_2_weight")
    fixed_net_params["conv3_1_weight"] = fix_weights_order(net_params, "conv3_1_weight")
    fixed_net_params["conv3_2_weight"] = fix_weights_order(net_params, "conv3_2_weight")
    fixed_net_params["conv3_3_weight"] = fix_weights_order(net_params, "conv3_3_weight")
    fixed_net_params["conv4_1_weight"] = fix_weights_order(net_params, "conv4_1_weight")
    fixed_net_params["conv4_2_weight"] = fix_weights_order(net_params, "conv4_2_weight")
    fixed_net_params["conv4_3_weight"] = fix_weights_order(net_params, "conv4_3_weight")
    fixed_net_params["conv5_1_weight"] = fix_weights_order(net_params, "conv5_1_weight")
    fixed_net_params["conv5_2_weight"] = fix_weights_order(net_params, "conv5_2_weight")
    fixed_net_params["conv5_3_weight"] = fix_weights_order(net_params, "conv5_3_weight")

    # Load the alpha_beta weights from a .mat file
    weights = matread(alpha_beta_file)
    
    return net_params, weights
end

# DISTS function to compute perceptual quality score between two images
function DISTS_score(ref, dist, net_params, weights, resize_img, use_gpu; pretraining = "DISTS")
    ref_features = extract_features(ref, net_params, resize_img, use_gpu; pretraining = pretraining)
    dist_features = extract_features(dist, net_params, resize_img, use_gpu; pretraining = pretraining)

    dist1 = 0.0
    dist2 = 0.0
    c1 = 1e-6
    c2 = 1e-6
    chns = [3, 64, 128, 256, 512, 512]

    alpha = split_weights(weights["alpha"], chns)
    beta = split_weights(weights["beta"], chns)

    for i in 1:6
        ref_mean = mean(ref_features[i], dims=(1, 2))
        dist_mean = mean(dist_features[i], dims=(1, 2))

        ref_var = mean((ref_features[i] .- ref_mean).^2, dims=(1, 2))
        dist_var = mean((dist_features[i] .- dist_mean).^2, dims=(1, 2))

        ref_dist_cov = mean(ref_features[i] .* dist_features[i], dims=(1, 2)) .- ref_mean .* dist_mean

        S1 = (2 * ref_mean .* dist_mean .+ c1) ./ (ref_mean.^2 .+ dist_mean.^2 .+ c1)
        S2 = (2 * ref_dist_cov .+ c2) ./ (ref_var .+ dist_var .+ c2)

        dist1 += sum(alpha[i] .* vec(S1))
        dist2 += sum(beta[i] .* vec(S2))
    end
    score = 1 - (dist1 + dist2)
    return score
end

# Conversion from RGB object to Float32 and ignoring fourth channel in the PNG case
function rgb2matrix(image)
    ref_img = permutedims(channelview(image),[2,3,1])[:, :, 1:3]
    ref_img = Float32.(ref_img)
    return ref_img
end

# Feature extraction using a pretrained VGG16 network
function extract_features(I, layer_params, resize_img, use_gpu; pretraining="Flux")
    if resize_img && min(size(I)[1], size(I)[2]) > 256
        I = imresize(I, ratio=256 / min(size(I)[1], size(I)[2]))
    end

    I = rgb2matrix(I)
    # This is not fully implemented, so use_gpu is disabled
    if use_gpu
        @warn "gpu acceleration is planned, but not implemented, results are the same to when `use_gpu=false`"
        #I|>gpu
    end

    features = Vector{Any}(undef, 6)

    features[1] = I

    dlX = (I .- layer_params["vgg_mean"]) ./ layer_params["vgg_std"]
    dlX = Flux.unsqueeze(dlX, 4)

    if pretraining=="Flux"
        @info "Using Flux's Metalhead pretrained version of VGG16"
        vgg_model = VGG(16; pretrain=true)

        count = 1

        dlX = (I .- layer_params["vgg_mean"]) ./ layer_params["vgg_std"]
        dlX = Flux.unsqueeze(dlX, 4)
        
        # Stage 1
        dlY = vgg_model.layers[1][1](dlX)
        dlY = vgg_model.layers[1][2](dlY)
        features[2] = dlY

        # Stage 2
        dlY = vgg_model.layers[1][3](dlY.^2)
        dlY = sqrt.(dlY)
        dlY = vgg_model.layers[1][4](dlY)
        dlY = vgg_model.layers[1][5](dlY)
        features[3] = dlY

        # Stage 3
        dlY = vgg_model.layers[1][6](dlY.^2)
        dlY = sqrt.(dlY)
        dlY = vgg_model.layers[1][7](dlY)
        dlY = vgg_model.layers[1][8](dlY)
        dlY = vgg_model.layers[1][9](dlY)
        features[4] = dlY

        # Stage 4
        dlY = vgg_model.layers[1][10](dlY.^2)
        dlY = sqrt.(dlY)
        dlY = vgg_model.layers[1][11](dlY)
        dlY = vgg_model.layers[1][12](dlY)
        dlY = vgg_model.layers[1][13](dlY)
        features[5] = dlY

        # Stage 5
        dlY = vgg_model.layers[1][14](dlY.^2)
        dlY = sqrt.(dlY)
        dlY = vgg_model.layers[1][15](dlY)
        dlY = vgg_model.layers[1][16](dlY)
        dlY = vgg_model.layers[1][17](dlY)
        features[6] = dlY
    elseif pretraining=="DISTS"
        @info "Using DISTS pretrained weights for MATLAB of VGG16"
        # Stage 1
        dlY = Conv(layer_params["conv1_1_weight"],layer_params["conv1_1_bias"][1,:], stride=1, pad=1, relu)(dlX)
        dlY = Conv(layer_params["conv1_2_weight"],layer_params["conv1_2_bias"][1,:], stride=1, pad=1, relu)(dlY)
        features[2] = dlY

        # Stage 2
        dlY = hcat(zeros(257, 1, 64, 1), vcat(zeros(1, 256, 64, 1), dlY))
        temp_layer = DepthwiseConv((3,3), 64=>64, bias=false, stride=2)
        for i=1:64
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_1"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)
        dlY = Conv(layer_params["conv2_1_weight"],layer_params["conv2_1_bias"][1,:], stride=1, pad=1, relu)(dlY)
        dlY = Conv(layer_params["conv2_2_weight"],layer_params["conv2_2_bias"][1,:], stride=1, pad=1, relu)(dlY)
        features[3] = dlY

        # Stage 3
        temp_layer = DepthwiseConv((3,3), 128=>128, bias=false, stride=2)
        dlY = hcat(zeros(129, 1, 128, 1), vcat(zeros(1, 128, 128, 1), dlY))
        for i=1:128
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_2"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)

        dlY = Conv(layer_params["conv3_1_weight"],layer_params["conv3_1_bias"][1,:], stride=1, pad=1, relu)(dlY)
        dlY = Conv(layer_params["conv3_2_weight"],layer_params["conv3_2_bias"][1,:], stride=1, pad=1, relu)(dlY)
        dlY = Conv(layer_params["conv3_3_weight"],layer_params["conv3_3_bias"][1,:], stride=1, pad=1, relu)(dlY)
        features[4] = dlY

        # Stage 4
        temp_layer = DepthwiseConv((3,3), 256=>256, bias=false, stride=2)
        dlY = hcat(zeros(65, 1, 256, 1), vcat(zeros(1, 64, 256, 1), dlY))
        for i=1:256
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_3"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)

        dlY = Conv(layer_params["conv4_1_weight"],layer_params["conv4_1_bias"][1,:], stride=1, pad=1, relu)(dlY)
        dlY = Conv(layer_params["conv4_2_weight"],layer_params["conv4_2_bias"][1,:], stride=1, pad=1, relu)(dlY)
        dlY = Conv(layer_params["conv4_3_weight"],layer_params["conv4_3_bias"][1,:], stride=1, pad=1, relu)(dlY)
        features[5] = dlY

        # Stage 5
        temp_layer = DepthwiseConv((3,3), 512=>512, bias=false, stride=2)
        dlY = hcat(zeros(33, 1, 512, 1), vcat(zeros(1, 32, 512, 1), dlY))
        for i=1:512
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_4"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)

        dlY = Conv(layer_params["conv5_1_weight"],layer_params["conv5_1_bias"][1,:], stride=1, pad=1, relu)(dlY)
        dlY = Conv(layer_params["conv5_2_weight"],layer_params["conv5_2_bias"][1,:], stride=1, pad=1, relu)(dlY)
        dlY = Conv(layer_params["conv5_3_weight"],layer_params["conv5_3_bias"][1,:], stride=1, pad=1, relu)(dlY)

        features[6] = dlY
    elseif pretraining=="Mixed"
        @info "Using DISTS pretrained weights for MATLAB and Metalhead's version of VGG16"
        vgg_model = VGG(16; pretrain=true)

        # Stage 1
        dlY = vgg_model.layers[1][1](dlX)
        dlY = vgg_model.layers[1][2](dlY)
        features[2] = dlY

        # Stage 2
        dlY = hcat(zeros(257, 1, 64, 1), vcat(zeros(1, 256, 64, 1), dlY))
        temp_layer = DepthwiseConv((3,3), 64=>64, bias=false, stride=2)
        for i=1:64
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_1"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)
        dlY = vgg_model.layers[1][4](dlY)
        dlY = vgg_model.layers[1][5](dlY)
        features[3] = dlY

        # Stage 3
        temp_layer = DepthwiseConv((3,3), 128=>128, bias=false, stride=2)
        dlY = hcat(zeros(129, 1, 128, 1), vcat(zeros(1, 128, 128, 1), dlY))
        for i=1:128
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_2"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)

        dlY = vgg_model.layers[1][7](dlY)
        dlY = vgg_model.layers[1][8](dlY)
        dlY = vgg_model.layers[1][9](dlY)
        features[4] = dlY

        # Stage 4
        temp_layer = DepthwiseConv((3,3), 256=>256, bias=false, stride=2)
        dlY = hcat(zeros(65, 1, 256, 1), vcat(zeros(1, 64, 256, 1), dlY))
        for i=1:256
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_3"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)

        dlY = vgg_model.layers[1][11](dlY)
        dlY = vgg_model.layers[1][12](dlY)
        dlY = vgg_model.layers[1][13](dlY)
        features[5] = dlY

        # Stage 5
        temp_layer = DepthwiseConv((3,3), 512=>512, bias=false, stride=2)
        dlY = hcat(zeros(33, 1, 512, 1), vcat(zeros(1, 32, 512, 1), dlY))
        for i=1:512
            temp_layer.weight[:,:,:,i] = layer_params["L2pool_4"][:,:,1,:,i]
        end
        dlY = temp_layer(dlY.^2)
        dlY = sqrt.(dlY)

        dlY = vgg_model.layers[1][15](dlY)
        dlY = vgg_model.layers[1][16](dlY)
        dlY = vgg_model.layers[1][17](dlY)

        features[6] = dlY
    else
        @warn "Non-supported pretraining selection. Features are filled with undefined values."
    end
    return features
end

# Function to split weights based on channel sizes
function split_weights(w, chns)
    w_ = Vector{Any}(undef, length(chns))
    idx = 1
    for i in 1:length(chns)
        w_[i] = w[idx:idx + chns[i] - 1]
        idx += chns[i]
    end
    return w_
end

export DISTS_score, load_weights