from PIL import Image
from io import BytesIO
import requests
import numpy as np

from torchvision import transforms
import torch

def get_image_http(image_path):
    try:
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except:
        "Download problem!"

def get_image_local(image_path):
    try:
        return Image.open(image_path).convert('RGB')
    except:
        "Open file problem"

def load_image(image_path, max_size=500,
               shape=None, net=False,
               transform=None):
    '''
    Load image from source.

    Args:
        image_path: local file path or www file patch
        max_size: maximum of picture size, default 500
        shape: shape of picture, default None
        net: Is picture from net, default False
        transform: PyTorch tranform.Compose object, default None
    '''
    if net:
        image = get_image_http(image_path)
    else:
        image = get_image_local(image_path)

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    if transform is None:
        transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    image = transform(image)
    image = image[:3,:,:]
    image = image.unsqueeze(0)
    return image

def image_convert(tensor):
    """
    Convert a tensor to Numpy

    Args:
        tensor: PyTorch tensor object.
    """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def get_features(image, model, layers=None):
    """ 
    Run an image forward through a model and get the features for 
    a set of layers. 
        
    Args:
        image: PyTorch tensor representing image
        model: PyTorch model
        layer: layers 
    """
    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  #<-- content
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram 