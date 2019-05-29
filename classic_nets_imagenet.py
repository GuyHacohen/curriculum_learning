# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
import numpy as np
from keras import backend as K
#import datasets.cifar100_subset
import PIL.Image


def classify_img(input_images, input_model):
    num_images, rows, cols, channels = input_images.shape
    # define a dictionary that maps model names to their classes
    # inside Keras
    MODELS = {
        "vgg16": VGG16,
        "vgg19": VGG19,
        "inception": InceptionV3,
        "xception": Xception,  # TensorFlow ONLY
        "resnet": ResNet50
    }

    if input_model not in MODELS.keys():
        print("unsupported imagenet network")
        raise ValueError
    # initialize the input image shape (224x224 pixels) along with
    # the pre-processing function (this might need to be changed
    # based on which model we use to classify our image)
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

    # if we are using the InceptionV3 or Xception networks, then we
    # need to set the input shape to (299x299) [rather than (224x224)]
    # and use a different image processing function
    if input_model in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input

    print("[INFO] loading {}...".format(input_model))
    Network = MODELS[input_model]
    model = Network(weights="imagenet")

    # with a Sequential model
    get_last_layer_output = K.function([model.layers[0].input],
                                      [model.layers[-2].output])

    # load the input image using the Keras helper utility while ensuring
    # the image is resized to `inputShape`, the required input dimensions
    # for the ImageNet pre-trained network
    # print("[INFO] loading and pre-processing image...")
    # image = load_img(input_image, target_size=inputShape)
    # image = img_to_array(image)

    # our input image is now represented as a NumPy array of shape
    # (inputShape[0], inputShape[1], 3) however we need to expand the
    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
    # so we can pass it through the network
    # image = np.expand_dims(image, axis=0)

    # pre-process the image using the appropriate function based on the
    # model that has been loaded (i.e., mean subtraction, scaling, etc.)

    transfer_values = np.zeros((num_images, model.layers[-2].output_shape[1]))
    for i in range(num_images):
        print("transferring image: " + str(i) + r"/" + str(num_images), end='\r')
        image = PIL.Image.fromarray(input_images[i, :, :, :]).resize(inputShape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess(image)
        # model.predict(images)
        # classify the image
        # print("[INFO] classifying images with '{}'...".format(input_model))
        transfer_value = get_last_layer_output([image])[0]
        transfer_values[i, :] = transfer_value
    print("transferring image: " + str(num_images) + r"/" + str(num_images))
    return transfer_values


# dataset = cifar100_subset.Cifar100_Subset(supeclass_idx=16, normalize=False)

# classify_img(dataset.x_train, 'vgg16')
# classify_img(dataset.x_train, 'vgg19')
# classify_img(dataset.x_train, 'inception')
# classify_img(dataset.x_train, 'xception')
# classify_img(dataset.x_train, 'resnet')