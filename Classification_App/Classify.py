import json

import torch
from PIL import Image
import Model
import numpy as np
import re

model_afib = None
model_iavb = None
model_lbbb = None
model_rbbb = None
model_pac = None
model_pvc = None
model_std = None
model_ste = None

model_dict = {"afib": model_afib,
              "iavb": model_iavb,
              "lbbb": model_lbbb,
              "rbbb": model_rbbb,
              "pac": model_pac,
              "pvc": model_pvc,
              "std": model_std,
              "ste": model_ste
              }

checkpoint_dict = {"afib": r"./checkpoints/checkpoint_type_afib.pt",
                   "iavb": r"./checkpoints/checkpoint_type_iavb.pt",
                   "lbbb": r"./checkpoints/checkpoint_type_lbbb.pt",
                   "rbbb": r"./checkpoints/checkpoint_type_rbbb.pt",
                   "pac": r"./checkpoints/checkpoint_type_pac.pt",
                   "pvc": r"./checkpoints/checkpoint_type_pvc.pt",
                   "std": r"./checkpoints/checkpoint_type_std.pt",
                   "ste": r"./checkpoints/checkpoint_type_ste.pt"}


def loadModel(checkpoint_type):
    torch.multiprocessing.freeze_support()
    device = torch.device('cpu')

    # !Defining the model!

    # General params
    in_channels = 3 # rgb pixels
    in_h = 675 # these 2 match the picture dimension
    in_w = 1450

    # num of channels and kernel length in each layer, note that list lengths must correspond
    hidden_channels = [8, 16, 32, 64, 128, 256, 512] #layers
    kernel_sizes = [5] * 7 # cnn kernel size

    # which tricks to use: dropout, stride, batch normalization and dilation
    dropout = 0.2 
    stride = 2
    dilation = 1
    batch_norm = True

    # fully connected net structure:

    # num of hidden units in every fully connected layer
    fc_hidden_dims = [128]

    # num of output classes
    num_of_classes = 2

    model = Model.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
                                fc_hidden_dims, dropout=0.2, stride=stride,
                                dilation=dilation, batch_norm=batch_norm, num_of_classes=2).to(device)

    print(f"Loading checkpoint for model for {checkpoint_type}")
    checkpoint_filename = checkpoint_dict[checkpoint_type]
    saved_state = torch.load(checkpoint_filename, map_location=device)
    model.load_state_dict(saved_state['model_state'])
    model.train(False)
    return model


def testImage(image, model, checkpoint_type):
    A = image
    img = A.resize((1450, 675), Image.ANTIALIAS)
    Image_to_test = torch.from_numpy(np.array(img)).float().unsqueeze(0)
    K = Image_to_test.permute(0, 3, 1, 2).to('cpu')
    out = model(K)
    tensor = out.item()
    classification = False
    print(f'Out: {out}')
    if tensor <= 0:
        print(f"Not {checkpoint_type}")
    elif tensor > 0:
        print(f"Is {checkpoint_type}")
        classification = True

    return checkpoint_type, classification


def load_models():
    for key in model_dict.keys():
        model_dict[key] = loadModel(key)


def predict(image):
    result_dict = {'results': []}
    for key in model_dict.keys():
        print(f"classifying checkpoint: {key}")
        current_res = testImage(image, model_dict[key], key)
        result_dict['results'].append({'classification': current_res[0], 'result': current_res[1]})
    final_res = json.dumps(result_dict)
    print(final_res)
    return result_dict
