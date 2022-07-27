# Imports
from __future__ import print_function
import transforms as tf
from transforms import *
from ECG_dataloader import *
import models
from PIL import Image
import pandas as pd
import glob

'''
- normal sinus
or
- one or more of the following disorder types:       index:
    *atrial fibrillation(AF)                           1
    *first - degree atrioventricular block(I - AVB)    2
    *left bundle branch block(LBBB)                    3
    *right bundle branch block(RBBB)                   4
    *premature atrial contraction(PAC)                 5
    *premature ventricular contraction(PVC)            6
    *ST - segment depression(STD)                      7
    *ST - segment elevation(STE)                       8
'''


def test_my_image(model, path):
    img_path = Image.open(path)
    img = img_path.resize((1450, 675), Image.ANTIALIAS)
    Image_to_classify = torch.from_numpy(np.array(img)).float().unsqueeze(0)
    K = Image_to_classify.permute(0, 3, 1, 2).to('cuda')
    out = model(K)
    return out


# TODO explain this method
def tensor_results_to_df(model):
    l2 = []
    path = None  # somePath
    paths = glob.glob(path)
    for i in range(1, 30):
        x = test_my_image(model, path=paths[i]).item()
        l2.append(x)

    n2 = np.array(l2)
    df = pd.DataFrame(n2)
    return df


def ECG_Image_Classification(
        perspective_transform=False,
        Is_classifier=False,
        Image_to_classify=None,
        classification_threshold=None, GPU_num=0, class_to_train=None):
    """
        Driver method to train and create the model.
        :param perspective_transform: Future implementation - Train to fix image rotation.
        :param Is_classifier: Whether or not to check on an image after training.
        :param Image_to_classify: If Is_classifier, which image to try
    """

    # paths
    torch.multiprocessing.freeze_support()
    # Path to save / load checkpoints (Weights,Biases) from.
    checkpoints_name = r'C:\Users\Kotler\PycharmProjects\ECG\ECG4U\Training_And_Classification_App\checkpoints\checkpoint_type_RBBB'
    # apply_perspective_transformation = perspective_transform
    apply_perspective_transformation = False

    # Batch size
    if apply_perspective_transformation:
        batch_size = 30
        # batch_size = 4
    else:
        batch_size = 100
        # batch_size = 4
    print("batch size: %s" % batch_size)
    # Data sizes for training

    # for real training:
    num_train = 35000
    num_val = 1000
    num_test = 5830
    # for small set:
    # num_train = 4
    # num_val = 4
    # num_test = 4

    # Device definition (cuda/cpu)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # torch.cuda.set_device(device=GPU_num)
    print('Using device:', device)

    # Creating a wrapper for the entire Database before dividing and converting it to specific Datasets and DataLoader
    ds = ECG_Rendered_Multilead_Dataset(apply_perspective_transformation=apply_perspective_transformation,class_to_train = 4)

    # ~Training~ dataset & loader
    ds_train = tf.SubsetDataset(ds, num_train)  # (train=True, transform=tf_ds)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    x, y = next(iter(dl_train))

    # ~Validation~ dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size, num_workers=2, pin_memory=True)

    # ~Test~ dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size, num_workers=2, pin_memory=True)

    x, y = iter(dl_train).next()

    # Architecture definition
    # General params
    in_h = x.shape[1]
    in_w = x.shape[2]
    in_channels = x.shape[3]
    batch_memory = x.element_size() * x.nelement() // 1024 ** 2
    x = x.transpose(1, 2).transpose(1, 3)

    # n channels and kernel length in each layer
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [5] * 7

    # which tricks to use: dropout, stride, batch normalization and dilation
    dropout = 0.2  # only for training! not for test!
    stride = 2
    dilation = 1
    batch_norm = True

    # FC net structure:
    fc_hidden_dims = [128]  # num of hidden units in every FC layer
    num_of_classes = 2  # num of output classess

    # Define the model
    model = models.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=2).to(device)

    # %% Test the dimentionality
    x_try = x.to(device, dtype=torch.float)
    y_pred = model(x_try)
    print('Output batch size is:', y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')

    num_correct = torch.sum((y_pred > 0).flatten() == (
            y.to(device, dtype=torch.long) == 1))
    # print(100*num_correct.item()/len(y),
    #       '% Accuracy... maybe we should consider training the model')

    del x, y, x_try, y_pred

    # %% Let's start training
    import torch.nn as nn
    import torch.optim as optim
    from training import Ecg12LeadImageNetTrainerBinary

    torch.manual_seed(42)

    lr = 0.001
    checkpoint_filename = f'{checkpoints_name}.pt'
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    if os.path.isfile(path + '//' + checkpoint_filename):
        num_epochs = 200
    else:
        # num_epochs = 30
        num_epochs = 200


    ## to check wtf is the accuracy
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Ecg12LeadImageNetTrainerBinary(model, loss_fn, optimizer, device,
                                             classification_threshold=classification_threshold)

    # fitResult2 = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=checkpoints_name,  # dl_val
    #                          early_stopping=5, print_every=1)
    fitResult2 = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=checkpoints_name,  # dl_val
                             print_every=1)
    if Is_classifier:
        my_image = test_my_image(model)
        print(f'Out: {my_image}')

    #####################   ROC #################################################
    thresholds = np.arange(0, 1, 0.01, dtype=float)
    for th in thresholds:
        trainer.classification_threshold = th
        test_result = trainer.test_epoch(dl_test, verbose=True)
        with open(f'ROC_Image_Pers_{perspective_transform}.txt', "a") as myfile:
            myfile.write(
                f'{th}\t{test_result.num_TP}\t{test_result.num_TN}\t{test_result.num_FP}\t{test_result.num_FN}\t{test_result.accuracy}\n')
    trainer.classification_threshold = None
    #####################   END OF ROC  #################################################

    test_result = trainer.test_epoch(dl_test, verbose=True)
    saved_state = dict(model_state= model.state_dict())
    torch.save(saved_state, checkpoint_filename)
    print('Test accuracy is: ', test_result[1], '%')

    return test_result


# Main loop
if __name__ == "__main__":
    print('Start training')
    perspective_transform = False
    test_results = ECG_Image_Classification(
        perspective_transform=perspective_transform,
        classification_threshold=None,
        GPU_num=0,
        class_to_train = 4
    )
    # 1 print('Train Image to class with perspective transformation') perspective_transform = True test_results =
    # 1 ECG_Image_Classification(perspective_transform=perspective_transform, classification_threshold=None, GPU_num=0)

    # 2 for class_type in range(9):
    # 2     with open("Execution_dump.txt", "a") as myfile:
    # 2         myfile.write(f'Executing class number: {class_type}\n')
    # 2         #Classifier for each type(class_type=class_type)

    print("hello world")
