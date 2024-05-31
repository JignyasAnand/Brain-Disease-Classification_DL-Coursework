import os

from Project.vgNet import vgNet
from preprocess import structure_datasets, get_ds_splits
from ANN import ann
from CNN import cnn

if __name__ == "__main__":
    base_directory = r'D:\KL\KL-3rd yr\Deep Learning\Data Set'

    structure_datasets(base_directory)
    train_generator, test_generator = get_ds_splits("Brain Stroke",base_directory)
    ann(train_generator,test_generator)
    #cnn(train_generator, test_generator)
    #vgNet(train_generator, test_generator)

#print("Image preprocessing completed.")
