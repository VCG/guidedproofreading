from lasagne import layers
from lasagne import nonlinearities

from cnn import CNN

class RGBNetPlus(CNN):
    '''
    Our CNN with image, prob, merged_array as RGB.

    This includes dropout. This also includes more layers.
    '''

    def __init__(self):

        CNN.__init__(self,

            layers=[
                ('input', layers.InputLayer),

                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),

                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),

                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('dropout3', layers.DropoutLayer),

                ('conv4', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
                ('dropout4', layers.DropoutLayer),                

                ('hidden5', layers.DenseLayer),
                ('dropout5', layers.DropoutLayer),
                ('output', layers.DenseLayer),
            ],

            # input
            input_shape=(None, 3, 75, 75),

            # conv2d + pool + dropout
            conv1_filter_size=(3,3), conv1_num_filters=64,
            pool1_pool_size=(2,2),
            dropout1_p=0.2,

            # conv2d + pool + dropout
            conv2_filter_size=(3,3), conv2_num_filters=48,
            pool2_pool_size=(2,2),
            dropout2_p=0.2,

            # conv2d + pool + dropout
            conv3_filter_size=(3,3), conv3_num_filters=48,
            pool3_pool_size=(2,2),
            dropout3_p=0.2,

            # conv2d + pool + dropout
            conv4_filter_size=(3,3), conv4_num_filters=48,
            pool4_pool_size=(2,2),
            dropout4_p=0.2,            

            # dense layer 1
            hidden5_num_units=512,
            hidden5_nonlinearity=nonlinearities.rectify,
            dropout5_p=0.5,

            # dense layer 2
            output_num_units=2,
            output_nonlinearity=nonlinearities.softmax

        )
