import sys
import time

import numpy as np
import matplotlib.image as plt_img
import torch
import torch.optim as optim
from torchvision import models
from tqdm import trange

import function_helper as fh

def load_data(style_path, content_path, device):
    content = fh.load_image(content_path,
                            max_size=300).to(device)
    style = fh.load_image(style_path,
                          shape=content.shape[-2:]).to(device)

    return content, style

def save_image(target_path, target):
    plt_img.imsave(target_path, fh.image_convert(target))

def main(style_path, content_path, target_path, steps=3800):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg = models.vgg19(pretrained=True).features
    vgg.to(device)

    for param in vgg.parameters():
        param.requires_grad_(False)

    content, style = load_data(style_path, content_path, device)

    content_features = fh.get_features(content, vgg)
    style_features = fh.get_features(style, vgg)

    style_grams = {layer: fh.gram_matrix(style_features[layer])
                   for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    # initialize weight for network layers
    style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    optimizer = optim.Adam([target], lr=0.003)
    
    if steps is None:
        steps = 3800

    #for ii in range(1, steps+1):
    for ii in trange(steps):
        
        # get the features from your target image
        target_features = fh.get_features(target, vgg)
        
        # the content loss
        content_loss = torch.mean((target_features['conv4_2'] 
                                    - content_features['conv4_2'])**2)
        
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = fh.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * \
                                torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
            
        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        time.sleep(0.01)
    
    save_image(target_path, target)

if __name__ =="__main__":
    if len(sys.argv) < 4:
        print(f"Use: python main.py path/to/content_image.jpg "
                f"path/to/style_image,jpg path/to/target.jpg")
        sys.exit(-1)
    elif len(sys.argv) == 5:
        main(sys.argv[2], sys.argv[1], sys.argv[3], steps=int(sys.argv[4]))
    else:
        main(sys.argv[2], sys.argv[1], sys.argv[3])

    
