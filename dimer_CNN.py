import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras import backend

#there are 4 labels: a_p (radius), z_p (axial distance), theta (polar angle), and phi (azimuthal angle).
# all 4 labels are normalized [0,1]

def rsq(y_true, y_pred):
    '''calculate coefficient of determination for regression accuracy'''
    ss_res = backend.sum(backend.square(y_true - y_pred))
    ss_tot = backend.sum((backend.square(y_true - backend.mean(y_true))))
    return (1 - ss_res/(ss_tot + backend.epsilon()))

def run_CNN(load_weights):
    '''run the CNN by either building the network, or by loading existing model'''

    #load data
    print('loading data...')
    parent = '/scratch/ja3067/mydata/'
    X = np.load(parent + 'X.npy')
    Y = np.load(parent + 'Y.npy')
    print('...loading data completed')

    if load_weights == True:
        model = load_model("save_weights01.hdf5", custom_objects={"rsq": rsq})

    else:
        #build the model:
        
        X = X.reshape(11000, 301, 301, 1) #change accordingly

        model = Sequential() 

        #add model layers (5 layers and then 3). Pool after every convolution
        model.add(Conv2D(16, kernel_size=5, input_shape=(301,301,1))) #input is 301x301 (1 for grayscale)
        model.add(Activation("relu"))
        model.add(MaxPooling2D(strides=(2,2), padding='same'))

        model.add(Conv2D(32, kernel_size=3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(strides=(2,2), padding='same'))

        model.add(Conv2D(64, kernel_size=3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(strides=(2,2), padding='same'))

        model.add(Conv2D(128, kernel_size=3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(strides=(2,2), padding='same'))

        model.add(Dropout(0.25)) #to prevent overfitting
        #flatten
        model.add(Flatten())

        #add fully connected layer
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(4, activation='linear')) #linear because it is a regression problem

        sgd = optimizers.legacy.SGD(learning_rate=.1, decay=1e-2, momentum=.9) #in case we want
            #SGD, instead of ADAM
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rsq])

        model.save('model01_noisy.h5')

    model.summary()

    csv_logger =  CSVLogger('training01.log')
    checkpoint = ModelCheckpoint("save_weights01.hdf5", monitor='loss', verbose=1,\
                             save_best_only=True, mode='auto')
    #shuffle data
    (X_train, X_val, Y_train, Y_val) = train_test_split(X, Y, test_size = .2, \
                                                        random_state = 42)
    model.fit(X_train, Y_train, batch_size = 16, epochs=3000,\
              validation_data = (X_val, Y_val), shuffle=True, callbacks=[csv_logger, checkpoint])

if __name__ == "__main__":

    run_CNN(load_weights=False)
