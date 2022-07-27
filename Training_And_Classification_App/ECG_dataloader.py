from torch.utils.data import Dataset
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
# from  Perspective_transformation import *
# from  Realtime_ECG_drawing import *
import time
import random
import cv2
import zipfile



class ECG_Rendered_Multilead_Dataset(Dataset):
    """
        Class to wrap the entire DB in the from (ImageData,Labels).
        Inherits from pytorch Dataset
    """
    # Convention   [n , height, width, color channel] 
    def __init__(self, root_dir=None, transform=None, partial_upload=False, new_format=True,
                 apply_perspective_transformation=False, class_to_train=None):
        super().__init__()
        self.data = []
        self.data_info = []
        self.transform = transform
        self.partial_upload = partial_upload
        self.classification_data = []
        self.root_dir = root_dir
        self.apply_perspective_transformation = apply_perspective_transformation

        if root_dir is None:
            self.dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Database')
        else:
            self.dataset_path = root_dir

        path_signal_classified_db = r'C:\Users\Kotler\PycharmProjects\ECG\ECG_Database\Data\Classified_Digitized_Signals.hdf5'
        classification_data = []
        f = h5py.File(path_signal_classified_db, 'r')
        f_keys = f.keys()
        for key in f_keys:
            n1 = f.get(key)
            classification_data.append(np.array(n1))

            """
                # Record in f[idx] holds the classification data for the 
                # ECG signal, and ECG image in the other file f2 in position f2[idx]
            """
        # The next part was commented out in order to test what happens if we train for a single type:
        if class_to_train == None:
            for batch_cntr in range(len(classification_data)):
                for record_in_batch_cntr in range(len(classification_data[batch_cntr])):
                    self.classification_data.append(bool(classification_data[batch_cntr][record_in_batch_cntr]))
        # [0,0,0,0,1,0,0,0], [1,0,0,0,0,0,0,0,0,0]:
            #   Before --> F, F, F, F, T, F, F, F, *T*, F, F, F, F, F
            #   After  --> F, F, F, F, T, F, F, F, *F*, F, F, F, F, F

        elif isinstance(class_to_train, int):
            for batch_cntr in range(len(classification_data)):
                for record_in_batch_cntr in range(len(classification_data[batch_cntr])):
                    if record_in_batch_cntr == class_to_train:
                        self.classification_data.append(bool(classification_data[batch_cntr][class_to_train]))
                    else:
                        self.classification_data.append(False)



    def __len__(self):
        return len(self.classification_data)

    def __getitem__(self, idx):
        with h5py.File(r'C:\Users\Kotler\PycharmProjects\ECG\ECG_Database\ImagesFromSignal.hdf5', "r") as f:
            n1 = f.get(str(idx))
            image_data = np.array(n1)
            # if self.apply_perspective_transformation:
            # image_data=Perspective_transformation_application(image_data,database_path=self.dataset_path)
        #The idx here is determined by self.offset+num_train/test/val
        # print(idx, self.classification_data[idx])
        sample = (image_data, self.classification_data[idx])
        """
            Returns the image (Pixilated) along with the corresponding
            classification from Classification_Data.
            The way both DBs are built is that the Signals and classification DB, matches the images DB in indexes.
        """
        return sample

    # def unpickle_ECG_data(self, file='ECG_data.pkl'):
    #     with open(file, 'rb') as fo:
    #         pickled_data = pickle.load(fo, encoding='bytes')
    #     print(f'Loaded data with type of: {type(pickled_data)}')
    #     return pickled_data

    def plot(self, idx):
        # TODO : Implement plot
        item_to_show = self.__getitem__(idx)
        plt.imshow(item_to_show[0])
        plt.show()
        return


if __name__ == "__main__":
    # New database directory
    ECG_test = ECG_Rendered_Multilead_Dataset(root_dir=None, transform=None, partial_upload=False,
                                              apply_perspective_transformation=False)  # For KNN demo
    testing_array = list(range(0, 10000))
    start = time.time()
    for indx in testing_array:
        rand_indx = random.randint(10, 41830)
        K = ECG_test[rand_indx]
        if (indx % 20 == 0) and indx > 0:
            print(f'Currently processing index: {indx}')
            end = time.time()
            print((end - start) / indx)
        # plt.imshow(K[0])
        # figManager = plt.get_current_fig_manager()
        # plt.show()
        print(f'Record: {indx} ,Is AFIB: {K[1]}')
    end = time.time()
    print(end - start)
