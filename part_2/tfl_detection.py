"""
Hands on lecture: train TFL net
In this hands-on lecture, we will train a model for predicting Traffic-light (TFL) in image patches,
you will use the data generated in previous course to this end.
You will :
1. Evaluate the data previously generated and ensure it's validity.
2. Train a CNN
3. Evaluate results of the trained model.
4. seek ways to improve results
"""

# %reload_ext autoreload
# %autoreload 2
# %autosave 120
# %matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from os.path import join

"""
Step 1. Validate your data
use the example in the cell below, to verify the TFL patch you've generated is sane.
Things to watch for:
1. You are able to load and vizualize your train and val data, using the functions below.
2. using the vizualization verify  image <--> label correspondence is correct.
3. % Negative vs. Positive examples is aprroximately 50%
"""


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Yes TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                      sharex=True, sharey=True)[1]  # .flatten()
    idxs = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)


# root = './'  #this is the root for your val and train datasets
data_dir = './data_dir/'
datasets = {
    'val': load_tfl_data(join(data_dir, 'val')),
    'train': load_tfl_data(join(data_dir, 'train')),
}
for k, v in datasets.items():
    print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

viz_my_data(num=(6, 6), **datasets['val'])


"""
define the model used for training
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Activation, \
    Conv2D


def tfl_model():
    input_shape = (81, 81, 3)

    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for i in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
    #     spatial_layer(1,32)
    #     spatial_layer(2,64)
    #     spatial_layer(2,96)

    #     spatial_layer(1,20)
    #     MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
    #     spatial_layer(1,64)
    #     conv_bn_relu(16,kernel_size=(3,3),input_shape=input_shape)
    #     spatial_layer(2,16)
    #     conv_bn_relu(32,kernel_size=(3,3),input_shape=input_shape)

    model.add(Flatten())
    dense_bn_relu(96)
    model.add(Dense(2, activation='softmax'))
    return model


m = tfl_model()
m.summary()

"""
train
"""

from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

data_dir = './data_dir/'
datasets = {
    'val': load_tfl_data(join(data_dir, 'val')),
    'train': load_tfl_data(join(data_dir, 'train')),
}
# prepare our model
m = tfl_model()
m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])

train, val = datasets['train'], datasets['val']
# train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model,
# by predicting the label of the images in the val dataset.
history = m.fit(train['images'], train['labels'], validation_data=(val['images'], val['labels']), epochs=2)

# compare train vs val acccuracy,
# why is val_accuracy not as good as train accuracy? are we overfitting?
epochs = history.history
epochs['train_acc'] = epochs['acc']
plt.figure(figsize=(10, 10))
for k in ['train_acc', 'val_acc']:
    plt.plot(range(len(epochs[k])), epochs[k], label=k)

plt.legend();


"""
evaluate and predict

Now thet we have a model we can use to predict results on the validation dataset.
1. What can say about example that fail prediction? can we find patterns that are common for failure cases?
"""

import seaborn as sbn

predictions = m.predict(val['images'])
sbn.distplot(predictions[:, 0]);

predicted_label = np.argmax(predictions, axis=-1)
print('accuracy:', np.mean(predicted_label == val['labels']))

viz_my_data(num=(6, 6), predictions=predictions[:, 1], **val);


"""
Seek ways to improve resutls
1. Try to play with diffferent models , increase / decrease the number of conv layers or number of filters.
you'll need to find a balanced model that is sufficiently large but minimzes overfit
    * try to enable tensorboard vizualization (see keras/ tensorboard doc) to help you carry the analysis
2. Data augmentation: feed your network with more examples by using data augmentation techniques:
such as horizontal image flip, noise, etc
"""

"""
Saving the model
After we trained our model and made predictions with it,
we will now want to save the architecture together with its learned weights in order for us to be able to use it
in the TFL manager. The architecture will be saved as a json, the weights in the h5 format:
"""

m.save("model.h5")

# If you want to make sure that this model can be used on different operating systems and different
# versions of keras or tensorflow, this is the better way to save. For this project the simpler
# method above should work fine.

# json_filename = 'model.json'
# h5_filename   = 'weights.h5'
# # create a json with the model architecture
# model_json = m.to_json()
# # save the json to disk
# with open(json_filename, 'w') as f:
#     f.write(model_json)
# # save the model's weights:
# m.save_weights(h5_filename)
# print(" ".join(["Model saved to", json_filename, h5_filename]))


"""
Loading the model
"""

from tensorflow.keras.models import load_model

loaded_model = load_model("model.h5")

# If you use the more robust method of saving above, this is how you load the model.

# import keras

# with open(json_filename, 'r') as j:
#     loaded_json = j.read()

# # load the model architecture:
# loaded_model = keras.models.model_from_json(loaded_json)
# #load the weights:
# loaded_model.load_weights(h5_filename)
# print(" ".join(["Model loaded from", json_filename, h5_filename]))

# code copied from the training evaluation:
l_predictions = loaded_model.predict(val['images'])
sbn.distplot(l_predictions[:, 0])

l_predicted_label = np.argmax(l_predictions, axis=-1)
print('accuracy:', np.mean(l_predicted_label == val['labels']))
