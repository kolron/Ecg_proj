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

model_dict = {"afib": model_afib}
""" ,
"iavb": model_iavb,
"lbbb": model_lbbb,
"rbbb": model_rbbb,
"pac": model_pac,
"pvc": model_pvc,
"std": model_std,
"ste": model_ste
}"""

checkpoint_dict = {"afib": r"./checkpoints/checkpoint_type_afib.pt"}
"""
,"iavb": r"/checkpoints/checkpoint_type_iavb.pt",
"lbbb": r"/checkpoints/checkpoint_type_lbbb.pt",
"rbbb": r"/checkpoints/checkpoint_type_rbbb.pt",
"pac": r"/checkpoints/checkpoint_type_pac.pt",
"pvc": r"/checkpoints/checkpoint_type_pvc.pt",
"std": r"/checkpoints/checkpoint_type_std.pt",
"ste": r/checkpoints/checkpoint_type_ste.pt}
"""



def loadModel(checkpoint_type):
    torch.multiprocessing.freeze_support()
    device = torch.device('cpu')


    # !Defining the model!

    # General params
    in_channels = 3
    in_h = 675
    in_w = 1450

    # num of channels and kernel length in each layer, note that list lengths must correspond
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [5] * 7

    # which tricks to use: dropout, stride, batch normalization and dilation
    dropout = 0.2  # Random layers for training, 0 when we test it
    stride = 2
    dilation = 1
    batch_norm = True

    # FC net structure:

    # num of hidden units in every FC layer
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

    return tensor, classification, checkpoint_type


def load_models():
    for key in model_dict.keys():
        model_dict[key] = loadModel(key)


def predict(image):
    result_dict = {}
    for key in model_dict.keys():
        print(f"classifying checkpoint: {key}")
        result_dict[key] = testImage(image, model_dict[key], key)
    print(result_dict)
    return result_dict










# def testImage_loop():
#     n = True
#     while n:
#         path = input("Please enter your image path here: ")
#         name = path[path.rfind("\\"):]
#         if os.path.isfile(path):
#             print(name)
#             testImage(path, loadModel())
#             print("========================================")
#             print("========================================")
#             choice = input("another? y/n: ")
#             if choice == "n":
#                 n = False
#         else:
#             print("File doesn't exist, check again.")

# # !Device definition!
# GPU_num = 0
# torch.multiprocessing.freeze_support()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device=GPU_num)
#
#
# # !Defining the model!
#
# # General params
# in_channels = 3
# in_h = 675
# in_w = 1450
#
# # num of channels and kernel length in each layer, note that list lengths must correspond
# hidden_channels = [8, 16, 32, 64, 128, 256, 512]
# kernel_sizes = [5] * 7
#
# # which tricks to use: dropout, stride, batch normalization and dilation
# dropout = 0.2  # Random layers for training, 0 when we test it
# stride = 2
# dilation = 1
# batch_norm = True
#
#
#
# # FC net structure:
#
# # num of hidden units in every FC layer
# fc_hidden_dims = [128]
#
# # num of output classes
# num_of_classes = 2
#
# model = Model.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
#                             fc_hidden_dims, dropout=dropout, stride=stride,
#                             dilation=dilation, batch_norm=batch_norm, num_of_classes=2).to(device)
#
# # Choosing and loading checkpoints


# TODO return JSON classification object to main.py for both classification and tensor
