import numpy as np
from tensorflow import keras
import h5py


class H5DataGenerator(keras.utils.Sequence):    
    '''Class based on following article: 
       https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''

    def __init__(self, h5_file_path, batch_size, dim, n_channels, n_classes, labels_dict=None, shuffle=True):
        """Data Generator based on a hdf5 file

        Args:
            h5_file_path (string): path to the file to stream
            batch_size (int): Size of the batch
            dim (tuple): Output shape dimension
            n_channels (int): Number of channel in pixel definition
            n_classes (int): Number of output class to search for
            labels_dict (dict, optional): On the output labels are binary hot encoded. 
                                          Therefor labels must be integer. 
                                          If labels are stored as str in hdf5 file, a dict could be passed to make the conversionself.
                                          eg: {'label_1': 0, 'label_2': 1, 'label_3': 2, 'label_4': 1} 
                                          Defaults to None.
            shuffle (bool, optional): Data can be shuffled between each epoch. Defaults to True.
        """        
        self.h5_file_path = h5_file_path
        self.dim = dim
        self.batch_size = batch_size
        self.labels_dict = labels_dict
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.nb_total_samples = self.get_file_length()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nb_total_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.indexes = np.random.permutation(self.nb_total_samples)
        else:
            self.indexes = np.arange(self.nb_total_samples)

    def __data_generation(self, indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        with h5py.File(name=f"{self.h5_file_path}",mode="r") as f:
            for in_batch_index, in_dataset_index in enumerate(indexes):
                # Store image
                X[in_batch_index, ] = f['image'][in_dataset_index].reshape(*self.dim,1)
                
                # Store category
                if self.labels_dict:
                  # convert label to int using labels_dict
                  key = f['category'][in_dataset_index]
                  print(f['category'].shape,in_dataset_index,key)
                  cat = self.labels_dict[key]
                else:
                  cat = f['category'][in_dataset_index]
                y[in_batch_index] = cat

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def get_file_length(self):
        length=0
        with h5py.File(name=f"{self.h5_file_path}",mode="r") as f:
          length = f['image'].shape[0]
        return length

if __name__ == "__main__":
    # example
    from tensorflow.keras.models import Sequential

    # Parameters
    params = {'dim': (28, 28),
              'batch_size': 64,
              'n_classes': 6,
              'n_channels': 1,
              'shuffle': True}

    # Datasets
    # path to an h5 file
    train_dataset_h5_file_path = ''
    val_dataset_h5_file_path = ''
    # a dictionary called labels
    # where for each ID of the dataset,
    # the associated label is given by labels[ID]
    # eg: {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    labels_dict = {}

    # Generators
    training_generator = H5DataGenerator(train_dataset_h5_file_path, labels_dict, **params)
    validation_generator = H5DataGenerator(val_dataset_h5_file_path, labels_dict, **params)

    # Design model
    model = Sequential()
    [...]  # Architecture
    model.compile()

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)
