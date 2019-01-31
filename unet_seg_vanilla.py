from __future__ import print_function
############### Imports ######################################################################################
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgba2rgb
from skimage.transform import resize
from scipy.misc import imshow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras import backend as K
import glob
from tensorflow.keras.models import load_model
##############################################################################################################


############### TRAINING DATA CONSTANTS ######################################################################
# TODO: make sure to replace path_to_data_dir with your own path
path_to_data_dir = "class2_data/"
image_name = "img.png"
mask_name = "cumulativeMask.png"

image_height_for_training = 256
image_width_for_training = 256
##############################################################################################################


############### READING THE DATA #############################################################################
def get_training_data(path_to_data_dir):
    """
    Find training and validation images in the given directory.
    :param path_to_data_dir: the absolute path to a data directory.
    :return:
            train_file_names: a list of training file names.
            val_file_names: a list of validation file names.
    """
    train_path = os.path.join(path_to_data_dir, "train")
    val_path = os.path.join(path_to_data_dir, "val")

    print("Trying to find training and validation data in folders:", train_path, val_path)

    train_file_names = glob.glob(train_path + "/*/" + image_name)
    val_file_names = glob.glob(val_path + "/*/" + image_name)

    # If this assert is reached, path_to_data_dir is probably wrong
    assert (len(train_file_names) > 0 and len(val_file_names) > 0)

    print("Found", len(train_file_names), "training files")
    print("Found", len(val_file_names), "validation files")

    return train_file_names, val_file_names


def load_image(path_to_image):
    """
    Loads png image from path_to_image to a numpy array.
    :param path_to_image: the absolute path to a png cell image.
    :return: a 3-dimensional numpy array of shape (height, width, 3) representing a RGB cell image.
    """
    img_array = imread(path_to_image)
    img_array_as_rgb = rgba2rgb(img_array).astype('float')

    return img_array_as_rgb


def preprocess_image(img_array):
    """
    Perform preprocessing of a cell image before feeding it to the model.
    :param img_array: a 3-dimensional numpy array of shape (height, width 3) representing a RGB cell image.
    :return: a processed version of the input image.
             Currently implemented: a resize of the input image to (256, 256, 3).
    """
    processed_img = resize(img_array, (image_height_for_training, image_width_for_training, 3), order=3)

    return processed_img


def load_mask(path_to_mask):
    """
    Loads png binary mask from path_to_mask to a numpy array.
    :param path_to_mask: the absolute path to a png binary mask image.
    :return: a 2-dimensional numpy array of shape (height, width) representing a binary cell mask.
    """
    mask_array = imread(path_to_mask, as_gray=True)

    return mask_array


def preprocess_mask(mask_array):
    """
    Perform preprocessing of a binary mask before feeding it to the model.
    :param mask_array: a 2-dimensional numpy array of shape (height, width) representing a a binary cell mask.
    :return: a processed version of the input mask.
             Currently implemented: a resize of the input image to (256, 256).
    """
    processed_mask = resize(mask_array, (image_height_for_training, image_width_for_training), order=1)

    return processed_mask

##############################################################################################################


############### DATA GENERATOR ###############################################################################
def generate_training_batches(relevant_file_names, batch_size):
    # This function is called twice: once for training and once for validation, and generates data for the model

    num_files = len(relevant_file_names)

    while True:
        # One iteration of the outer while loop == one epoch of the model.
        # The order of the files is randomized per epoch by the following line:
        random_index_order = np.random.permutation(num_files)

        # Go over list of files, feeding batch_size files to the model at a time
        for batch_index in range(0, num_files - batch_size + 1, batch_size):
            for b in range(0, batch_size):
                cur_ind = random_index_order[batch_index + b]

                # Obtain paths to single example + corresponding segmentation mask
                image_path = relevant_file_names[cur_ind]
                mask_path = image_path.replace(image_name, mask_name)

                # Load example + mask as numpy arrays:
                img_array = load_image(image_path)
                img_array = preprocess_image(img_array)

                mask_array = load_mask(mask_path)
                mask_array = preprocess_mask(mask_array)

                # For debugging purposes, show image + mask after loading
                # imshow(img_array)
                # imshow(mask_array)

                # Add channel to mask array (as required by Tensorflow, 'channels_last')
                # Image itself already has 3 channels so no expansion is required
                mask_array = mask_array[..., np.newaxis]

                # Add dimensions to both image and mask so that they can be concatenated along this axis:
                img_array = np.expand_dims(img_array, axis=0)
                mask_array = np.expand_dims(mask_array, axis=0)

                if b == 0:
                    # First example in batch
                    batch_img = img_array
                    batch_mask = mask_array
                else:
                    batch_img = np.concatenate((batch_img, img_array), axis=0)
                    batch_mask = np.concatenate((batch_mask, mask_array), axis=0)

            yield (batch_img, batch_mask)
            batch_img = None
            batch_mask = None
##############################################################################################################


############### LEARNING RATE DECAY ##########################################################################

def lr_fun(epoch_num):
    if epoch_num < 10:
        lr = 5e-4
    elif epoch_num >= 10 and epoch_num < 25:
        lr = 1e-5
    elif epoch_num > 25 and epoch_num < 75:
        lr = 1e-6
    else:
        lr = 1e-6
    return lr
##############################################################################################################


############### METRICS AND TRAINING LOSS ####################################################################
def dice_coef(y_true, y_pred):
    # calculating the DICE coefficient with a smoothing term
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    smooth = 1

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    # calculating the DICE loss function for GD
    return 1 - dice_coef(y_true, y_pred)


#############################################################################################################


################## MODEL ARCHITECTURE DEFINITION ############################################################


###### FUNCTIONS DEFINING U-NET BLOCKS ######
def convolution_block(input_layer, num_filters, kernel_size, activation_func):
    conv1 = Conv2D(filters=num_filters, kernel_size=kernel_size, activation=activation_func,
                   kernel_initializer='he_uniform', padding='same')(input_layer)
    conv2 = Conv2D(filters=num_filters, kernel_size=kernel_size, activation=activation_func,
                   kernel_initializer='he_uniform', padding='same')(conv1)

    return conv2


def conv_downsample_block(input_layer, num_filters, kernel_size, activation_func, max_pool_shape):
    pool = MaxPooling2D(pool_size=max_pool_shape)(input_layer)
    conv_block = convolution_block(pool, num_filters, kernel_size, activation_func)

    return conv_block


def conv_upsample_block(input_layer, skip_connection_layer, num_filters, kernel_size, activation_func,
                        upsampling_shape):
    upsampling = UpSampling2D(size=upsampling_shape)(input_layer)
    concatenate_skip = concatenate([upsampling, skip_connection_layer], axis=3)
    conv_block = convolution_block(concatenate_skip, num_filters, kernel_size, activation_func)

    return conv_block


###### U-NET ARCHITECTURE ######
def get_unet(img_rows, img_cols, first_layer_num_filters=32, num_classes=1):
    inputs = Input((img_rows, img_cols, 3))

    ###### ENCODER BRANCH ######
    first_conv_block = convolution_block(input_layer=inputs,
                                         num_filters=first_layer_num_filters,
                                         kernel_size=(3, 3),
                                         activation_func="relu")
    conv_ds_block_1 = conv_downsample_block(input_layer=first_conv_block,
                                            num_filters=first_layer_num_filters * 2,
                                            kernel_size=(3, 3),
                                            activation_func="relu", max_pool_shape=(2, 2))
    conv_ds_block_2 = conv_downsample_block(input_layer=conv_ds_block_1,
                                            num_filters=first_layer_num_filters * 4,
                                            kernel_size=(3, 3),
                                            activation_func="relu", max_pool_shape=(2, 2))
    conv_ds_block_3 = conv_downsample_block(input_layer=conv_ds_block_2,
                                            num_filters=first_layer_num_filters * 8,
                                            kernel_size=(3, 3),
                                            activation_func="relu", max_pool_shape=(2, 2))

    ##### BOTTOM OF U-SHAPE #####
    bottom_conv_block = conv_downsample_block(input_layer=conv_ds_block_3,
                                              num_filters=first_layer_num_filters * 16,
                                              kernel_size=(3, 3),
                                              activation_func="relu", max_pool_shape=(2, 2))

    ###### DECODER BRANCH ######
    conv_us_block_1 = conv_upsample_block(input_layer=bottom_conv_block, skip_connection_layer=conv_ds_block_3,
                                          num_filters=first_layer_num_filters * 8,
                                          kernel_size=(3, 3),
                                          activation_func="relu", upsampling_shape=(2, 2))
    conv_us_block_2 = conv_upsample_block(input_layer=conv_us_block_1, skip_connection_layer=conv_ds_block_2,
                                          num_filters=first_layer_num_filters * 4,
                                          kernel_size=(3, 3),
                                          activation_func="relu", upsampling_shape=(2, 2))
    conv_us_block_3 = conv_upsample_block(input_layer=conv_us_block_2, skip_connection_layer=conv_ds_block_1,
                                          num_filters=first_layer_num_filters * 2,
                                          kernel_size=(3, 3),
                                          activation_func="relu", upsampling_shape=(2, 2))
    last_conv_block = conv_upsample_block(input_layer=conv_us_block_3, skip_connection_layer=first_conv_block,
                                          num_filters=first_layer_num_filters,
                                          kernel_size=(3, 3),
                                          activation_func="relu", upsampling_shape=(2, 2))

    output_layer = Conv2D(num_classes, (1, 1), activation='sigmoid')(last_conv_block)

    model = Model(inputs=[inputs], outputs=[output_layer])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=['accuracy'])
    print(model.summary())

    return model
#############################################################################################################


################## TRAINING THE MODEL #######################################################################
def create_and_train_model(training_data, validation_data, input_height, input_width, first_layer_num_filters,
                           num_classes, transfer_weights=""):
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet(input_height, input_width, first_layer_num_filters, num_classes)
    print(model.summary())
    try:
        model.load_weights(transfer_weights)
    except:
        print('Did not find existing model')

    model_checkpoint = ModelCheckpoint('cell_vanilla_unet_weights_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5',
                                       monitor='val_loss',
                                       save_best_only=False)
    model_tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)
    model_earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10)
    model_LRSchedule = LearningRateScheduler(lr_fun)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    batch_size = 8
    num_epochs = 30

    model.fit_generator(generate_training_batches(training_data, batch_size=batch_size),
                        steps_per_epoch=np.ceil(len(training_data) / batch_size), epochs=num_epochs, verbose=1,
                        validation_data=generate_training_batches(validation_data, batch_size=batch_size),
                        validation_steps=np.ceil(len(validation_data) / batch_size),
                        callbacks=[model_checkpoint, model_earlystop, model_LRSchedule, model_tensorboard])
    model.save('cell_vanilla_unet_weights.h5')
##############################################################################################################


################## DISPLAYING MODEL RESULTS ##################################################################
def postprocess_segmentation(img_array, segmentation_prediction):
    """
    Perform postprocessing of a segmentation prediction made by the model.
    :param img_array: a 3-dimensional numpy array of shape (height, width, 3) representing a RGB cell image.
    :param segmentation_prediction: the matching binary mask predicted by the model.
    :return: a processed version of the segmentation prediction.
            Currently implemented: a resize of the segmentation back to full image resolution.
    """
    processed_segmentation = resize(segmentation_prediction, (img_array.shape[0], img_array.shape[1]), order=3)

    processed_segmentation[processed_segmentation >= 0.5] = 1
    processed_segmentation[processed_segmentation < 0.5] = 0

    return processed_segmentation


def predict_single_image(path_to_image, trained_model, display_result=True):
    """
    Predict a cell segmentation for a single image.
    :param path_to_image: the absolute path to a cell png image.
    :param trained_model: a trained model to be used for prediction.
    :param display_result: a boolean flag indicating whether results should be displayed after prediction.
    :return: a 2-dimensional numpy array of shape (height, width) representing img_array's cell segmentation.
    """
    img_array = load_image(path_to_image)
    processed_img = preprocess_image(img_array)
    processed_img = np.expand_dims(processed_img, 0)

    segmentation_prediction = np.squeeze(trained_model.predict(processed_img))
    processed_segmentation = postprocess_segmentation(img_array, segmentation_prediction)

    if display_result:
        imshow(img_array)
        imshow(processed_segmentation)

    return processed_segmentation


def dice_coef_for_np_array(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    smooth = 1

    intersection_size = np.sum(y_true_flat * y_pred_flat)
    dice_score = (2. * intersection_size + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)

    return dice_score


def evaluate_model(path_to_folder, path_to_weights):
    """
    Predicts a cell segmentation for all images within a given folder (for example, the "val" folder).
    Each segmentation prediction will be save along its input image under the name "segmentationPrediction.png".
    In addition, an average dice score will be calculated over the given data set.
    :param path_to_folder: the absolute path to a folder containing cell png images, in the same structure as
            the "train" and "val" folders.
    :param path_to_weights: the absolute path to a .h5 weights file.
    :return: a floating number, representing the average dice score over the given data set.
    """
    # load trained model
    trained_model = get_unet(None, None, 32, 1)
    trained_model.load_weights(path_to_weights)

    # predict segmentation for all images in given folder
    average_dice_score = 0
    all_images_in_folder = glob.glob(path_to_folder + "/*/" + image_name)
    total_num_images = len(all_images_in_folder)

    for path_to_image in all_images_in_folder:
        print("Predicting segmentation for:", path_to_image)

        segmentation_prediction = predict_single_image(path_to_image, trained_model, display_result=False)

        save_path = os.path.join(os.path.dirname(path_to_image), "segmentationPrediction.png")
        imsave(save_path, segmentation_prediction)

        path_to_mask = path_to_image.replace(image_name, mask_name)
        mask_array = load_mask(path_to_mask)
        mask_array[mask_array > 0] = 1
        curr_dice_score = dice_coef_for_np_array(mask_array, segmentation_prediction)

        average_dice_score += curr_dice_score
        print("Dice score:", curr_dice_score)

    average_dice_score /= total_num_images
    return average_dice_score


################## RUNNING OUR CODE ##########################################################################
if __name__ == '__main__':
    # Set channel ordering to match Tensorflow backend: ('channels_last' for Tensorflow, 'channels_first' for Theano)
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code

    # Get training and validation data:
    training_data, validation_data = get_training_data(path_to_data_dir)

    # Train model using data:
    create_and_train_model(training_data[:20], validation_data[:20], input_height=None, input_width=None,
                           first_layer_num_filters=32, num_classes=1)

    # dice_score = evaluate_model(path_to_folder="", path_to_weights="")
    # print("The dice score for evaulated dataset:", dice_score)

##############################################################################################################