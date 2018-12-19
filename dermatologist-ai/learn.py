import os
import cv2
import time

from keras import Model, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from imutils import paths
import numpy as np

# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

train_data_dir = 'data/train/'
validation_data_dir = 'data/valid/'
test_data_dir = 'data/test/'
img_width = 100
img_height = 100
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 20



def load_skin_data(image_paths):
    data = []
    labels = []
    for image_path in image_paths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(image_path)
        image = cv2.resize(image, (img_width, img_height))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = image_path.split(os.path.sep)[-2]
        # print(label)

        if label == "melanoma":
            labels.append((1, 0, 0))
        elif label == "nevus":
            labels.append((0, 0, 1))
        elif label == "seborrheic_keratosis":
            labels.append((0, 1, 0))

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels


def save_to_npy_files(directory, x_train, y_train):
    np.save(directory + 'x.npy', x_train)
    np.save(directory + 'y.npy', y_train)


def load_from_npy_files(directory):
    x = np.load(directory + 'x.npy')
    y = np.load(directory + 'y.npy')

    return x, y


def prepare_dataset():
    image_paths = sorted(list(paths.list_images(validation_data_dir)))
    (x_valid, y_valid) = load_skin_data(image_paths)
    save_to_npy_files(validation_data_dir, x_valid, y_valid)

    image_paths = sorted(list(paths.list_images(train_data_dir)))
    (x_train, y_train) = load_skin_data(image_paths)
    save_to_npy_files(train_data_dir, x_train, y_train)

    image_paths = sorted(list(paths.list_images(test_data_dir)))
    (x_test, y_test) = load_skin_data(image_paths)
    save_to_npy_files(test_data_dir, x_test, y_test)


if 0:
    prepare_dataset()

st = time.time()
(x_train, y_train) = load_from_npy_files(train_data_dir)
(x_valid, y_valid) = load_from_npy_files(validation_data_dir)
(x_test, y_test) = load_from_npy_files(test_data_dir)

print("load time: {}s".format(time.time() - st))

print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)

print("x_valid.shape", x_valid.shape)
print("y_valid.shape", y_valid.shape)

print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)

from keras.applications.vgg16 import VGG16
model = VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

model.summary()

for layer in model.layers[:5]:
    layer.trainable = False

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation

train_gen = ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest",
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        zoom_range = 0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

valid_gen = ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest",
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        zoom_range = 0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_gen.fit(x_train)
valid_gen.fit(x_valid)

filename = "cancer_classification.h5"
# Save the model according to the conditions
checkpoint = ModelCheckpoint(filename, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# validation_steps=validation_size//batch_size

# fits the model on batches with real-time data augmentation:
model_final.fit_generator(train_gen.flow(x_train, y_train, batch_size=32),
                    samples_per_epoch = nb_train_samples,
                    epochs = epochs,
                    validation_data = valid_gen.flow(x_valid, y_valid),
                    nb_val_samples = nb_validation_samples,
                    callbacks = [checkpoint],
                    steps_per_epoch=len(x_train) / 32)


model_final.load_weights(filename)

predictions = []
for feature in x_test:
    pred = model_final.predict(feature)
    predictions.append(pred)

predictions = np.asarray(predictions)
print(predictions.shape)

print(predictions[0])