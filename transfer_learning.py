import tensorflow as tf
import os
from matplotlib import pyplot as plt
import math
import cifar_utils
import collections
import functools
from keras_custom_callbacks import SimpleLogCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


# Some hyper-parameters:
batch_size  = 32            # Images per batch (reduce/increase according to the machine's capability)
num_epochs  = 300           # Max number of training epochs
random_seed = 42            # Seed for some random operations, for reproducibility
cifar_info = cifar_utils.get_info()
print(cifar_info)

# Number of classes:
num_classes = cifar_info.features['label'].num_classes

# Number of images:
num_train_imgs = cifar_info.splits['train'].num_examples
num_val_imgs = cifar_info.splits['test'].num_examples

train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
val_steps_per_epoch   = math.ceil(num_val_imgs / batch_size)

input_shape = [224, 224, 3]

train_cifar_dataset = cifar_utils.get_dataset(
    phase='train', batch_size=batch_size, num_epochs=num_epochs, shuffle=True,
    input_shape=input_shape, seed=random_seed)

val_cifar_dataset = cifar_utils.get_dataset(
    phase='test', batch_size=batch_size, num_epochs=1, shuffle=False,
    input_shape=input_shape, seed=random_seed)


resnet50_feature_extractor = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_shape=input_shape)
# resnet50_feature_extractor.summary()
frozen_layers, trainable_layers = [], []
for layer in resnet50_feature_extractor.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = False
        frozen_layers.append(layer.name)
    else:
        if len(layer.trainable_weights) > 0:
            # We list as "trainable" only the layers with trainable parameters.
            trainable_layers.append(layer.name)

log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\n\033[94m', '\033[92m'
log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
log_end_format = '\033[0m'

# Logging the lists of frozen/trainable layers:
print("{2}Layers we froze:{4} {0} ({3}total = {1}{4}).".format(
    frozen_layers, len(frozen_layers), log_begin_red, log_begin_bold, log_end_format))
print("\n{2}Layers which will be fine-tuned:{4} {0} ({3}total = {1}{4}).".format(
    trainable_layers, len(trainable_layers), log_begin_blue, log_begin_bold, log_end_format))

features = resnet50_feature_extractor.output
avg_pool = GlobalAveragePooling2D(data_format='channels_last')(features)
predictions = Dense(num_classes, activation='softmax')(avg_pool)

resnet50_freeze = Model(resnet50_feature_extractor.input, predictions)

metrics_to_print = collections.OrderedDict([("loss", "loss"), 
                                            ("v-loss", "val_loss"),
                                            ("acc", "acc"), 
                                            ("v-acc", "val_acc"),
                                            ("top5-acc", "top5_acc"), 
                                            ("v-top5-acc", "val_top5_acc")])

model_dir = './models/resnet_keras_app_freeze_all'
callbacks = [
    # Callback to interrupt the training if the validation loss/metrics stops improving for some epochs:
    tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc',
                                     restore_best_weights=True),
    # Callback to log the graph, losses and metrics into TensorBoard:
    tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
    # Callback to simply log metrics at the end of each epoch (saving space compared to verbose=1/2):
    SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=1),
    # Callback to save the model (e.g., every 5 epochs), specifying the epoch and val-loss in the filename:
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'), period=5)
]

# Compile:
optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
resnet50_freeze.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
    ])

# Train:
history_freeze = resnet50_freeze.fit(
    train_cifar_dataset,  epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
    validation_data=val_cifar_dataset, validation_steps=val_steps_per_epoch,
    verbose=0, callbacks=callbacks)

fig, ax = plt.subplots(3, 2, figsize=(15,10), sharex='col')
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("acc")
ax[1, 1].set_title("val-acc")
ax[2, 0].set_title("top5-acc")
ax[2, 1].set_title("val-top5-acc")

ax[0, 0].plot(history_freeze.history['loss'])
ax[0, 1].plot(history_freeze.history['val_loss'])
ax[1, 0].plot(history_freeze.history['acc'])
ax[1, 1].plot(history_freeze.history['val_acc'])
ax[2, 0].plot(history_freeze.history['top5_acc'])
ax[2, 1].plot(history_freeze.history['val_top5_acc'])

for layer in resnet50_feature_extractor.layers:
    if 'res5' in layer.name:
        # Keras developers named the layers in their ResNet implementation to explicitly 
        # identify which macro-block and block each layer belongs to.
        # If we reach a layer which has a name starting by 'resnet5', it means we reached 
        # the 4th macro-block / we are done with the 3rd one (see layer names listed previously):
        break
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = False

num_macroblocks_to_freeze = [0, 1, 2, 3] # we already covered the "all 4 frozen" case above.

histories = dict()
histories['freeze all'] = history_freeze
for freeze_num in num_macroblocks_to_freeze:
        
    print("{1}{2}>> {3}ResNet-50 with {0} macro-block(s) frozen{4}:".format(
        freeze_num, log_begin_green, log_begin_bold, log_begin_underline, log_end_format))
    
    # ---------------------
    # 1. We instantiate a new classifier each time:
    resnet50_feature_extractor = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', 
        input_shape=input_shape, classes=num_classes)

    features = resnet50_feature_extractor.output
    avg_pool = GlobalAveragePooling2D(data_format='channels_last')(features)
    predictions = Dense(num_classes, activation='softmax')(avg_pool)

    resnet50_finetune = Model(resnet50_feature_extractor.input, predictions)
    
    # ---------------------
    # 2. We freeze the desired layers: 
    break_layer_name = 'res{}'.format(freeze_num + 2) if freeze_num > 0 else 'conv1'
    frozen_layers = []
    for layer in resnet50_finetune.layers:
        if break_layer_name in layer.name:
            break
        if isinstance(layer, tf.keras.layers.Conv2D):
            # If the layer is a convolution, and isn't after the 1st layer not to train:
            layer.trainable = False
            frozen_layers.append(layer.name)
    
    print("\t> {2}Layers we froze:{4} {0} ({3}total = {1}{4}).".format(
        frozen_layers, len(frozen_layers), log_begin_red, log_begin_bold, log_end_format))
    
    # ---------------------
    # 3. To start from the beginning the data iteration, 
    #    we re-instantiate the input pipelines (same parameters):
    train_cifar_dataset = cifar_utils.get_dataset(
    phase='train', batch_size=batch_size, num_epochs=num_epochs, shuffle=True,
    input_shape=input_shape, seed=random_seed)

    val_cifar_dataset = cifar_utils.get_dataset(
        phase='test', batch_size=batch_size, num_epochs=1, shuffle=False,
        input_shape=input_shape, seed=random_seed)

    # ---------------------
    # 4. We set up the training operations, and start the process:
    # We set a smaller learning rate for the fine-tuning:
    # optimizer = tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)

    model_dir = './models/resnet_keras_app_freeze_{}_mb'.format(freeze_num)
    callbacks = [
        # Callback to interrupt the training if the validation loss/metrics converged:
        # (we use a shorter patience here, just to shorten a bit the demonstration, already quite long...)
        tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc', restore_best_weights=True),
        # Callback to log the graph, losses and metrics into TensorBoard:
        tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
        # Callback to save the model (e.g., every 5 epochs)::
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'), period=5)
    ]
    
    # Compile:
    resnet50_finetune.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
        ])

    # Train:
    print("\t> Training - {0}start{1} (logs = off)".format(log_begin_red, log_end_format))
    history = resnet50_finetune.fit(
        train_cifar_dataset,  epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
        validation_data=val_cifar_dataset, validation_steps=val_steps_per_epoch,
        verbose=0, callbacks=callbacks)
    print("\t> Training - {0}over{1}".format(log_begin_green, log_end_format))

    acc = history.history['acc'][-1] * 100
    top5 = history.history['top5_acc'][-1] * 100
    val_acc = history.history['val_acc'][-1] * 100
    val_top5 = history.history['val_top5_acc'][-1] * 100
    epochs = len(history.history['val_loss'])
    print("\t> Results after {5}{0}{6} epochs:\t{5}acc = {1:.2f}%; top5 = {2:.2f}%; val_acc = {3:.2f}%; val_top5 = {4:.2f}%{6}".format(
        epochs, acc, top5, val_acc, val_top5, log_begin_bold, log_end_format))

    histories['freeze {}'.format(freeze_num)] = history

fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex='col')
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("acc")
ax[1, 1].set_title("val-acc")
ax[2, 0].set_title("top5-acc")
ax[2, 1].set_title("val-top5-acc")

lines, labels = [], []
for config_name in histories:
    history = histories[config_name]
    ax[0, 0].plot(history.history['loss'])
    ax[0, 1].plot(history.history['val_loss'])
    ax[1, 0].plot(history.history['acc'])
    ax[1, 1].plot(history.history['val_acc'])
    ax[2, 0].plot(history.history['top5_acc'])
    line = ax[2, 1].plot(history.history['val_top5_acc'])
    lines.append(line[0])
    labels.append(config_name)

fig.legend(lines, labels, loc='center right', borderaxespad=0.1)
plt.subplots_adjust(right=0.87)

import glob
import numpy as np
from classification_utils import load_image, process_predictions, display_predictions

test_filenames = glob.glob(os.path.join('res', '*'))
test_images = np.asarray([load_image(file, size=input_shape[:2]) 
                          for file in test_filenames])

image_batch = test_images[:16]

# Our model was trained on CIFAR images, which originally are 32x32px. We scaled them up
# to 224x224px to train our model on, but this means the resulting images had important
# artifacts/low quality.
# To test on images of the same quality, we first resize them to 32x32px, then to the 
#expected input size (i.e., 224x224px):
cifar_original_image_size = cifar_info.features['image'].shape[:2]
class_readable_labels = cifar_info.features["label"].names

image_batch_low_quality = tf.image.resize(image_batch, cifar_original_image_size)
image_batch_low_quality = tf.image.resize(image_batch_low_quality, input_shape[:2])
    
predictions = resnet50_finetune.predict_on_batch(image_batch_low_quality)
top5_labels, top5_probabilities = process_predictions(predictions, class_readable_labels)

print("ResNet-50 trained on ImageNet and fine-tuned on CIFAR-100:")
display_predictions(image_batch, top5_labels, top5_probabilities)