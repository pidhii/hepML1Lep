## copied from A.Elwood https://github.com/aelwood/hepML/blob/master/MlFunctions/DnnFunctions.py

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import regularizers
from keras import backend as K
import keras
import math
import tensorflow as tf
import sys

def findLayerSize(layer,refSize):

    if isinstance(layer, float):
        return int(layer*refSize)
    elif isinstance(layer, int):
        return layer
    else:
        print('WARNING: layer must be int or float')
        return None
        
## copied from A.Elwood https://github.com/aelwood/hepML/blob/master/MlFunctions/DnnFunctions.py
def createDenseModel(inputSize=None,outputSize=None,hiddenLayers=[1.0],dropOut=None,l2Regularization=None,activation='relu',optimizer='adam',doRegression=False,loss=None,extraMetrics=[]):
    '''
    Dropout: choose percentage to be dropped out on each hidden layer (not currently applied to input layer)
    l2Regularization: choose lambda of the regularization (ie multiplier of the penalty)
    '''

    #check inputs are ok
    assert inputSize and outputSize, 'Must provide non-zero input and output sizes'
    assert len(hiddenLayers)>=1, 'Need at least one hidden layer'

    refSize=inputSize+outputSize

    #Initialise the model
    model = Sequential()

    if l2Regularization: 
        regularization=regularizers.l2(l2Regularization)
    else:
        regularization=None

    #Add the first layer, taking the inputs
    model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
        activation=activation, input_dim=inputSize,name='input',
        kernel_regularizer=regularization))


    if dropOut: model.add(Dropout(dropOut))

    #Add the extra hidden layers
    for layer in hiddenLayers[1:]:
        model.add(Dense(units=findLayerSize(hiddenLayers[0],refSize), 
            activation=activation,kernel_regularizer=regularization))

        if dropOut: model.add(Dropout(dropOut))

    if not doRegression: # if doing a normal classification model

        #Add the output layer and choose the type of loss function
        #Choose the loss function based on whether it's binary or not
        if outputSize==2: 
            #It's better to choose a sigmoid function and one output layer for binary
            # This is a special case of n=2 classification
            model.add(Dense(1, activation='sigmoid'))
            if not loss: loss = 'binary_crossentropy'
        else: 
            #Softmax forces the outputs to sum to 1 so the score on each node
            # can be interpreted as the probability of getting each class
            model.add(Dense(outputSize, activation='softmax'))
            if not loss: loss = 'categorical_crossentropy'

        #After the layers are added compile the model
        model.compile(loss=loss,
            optimizer=optimizer,metrics=['accuracy']+extraMetrics)

    else: # if training a regression add output layer with linear activation function and mse loss

        model.add(Dense(1))
        if not loss: loss='mean_squared_error'
        model.compile(loss=loss,
            optimizer=optimizer,metrics=['mean_squared_error']+extraMetrics)


    return model

def significanceLoss(expectedSignal,expectedBkgd):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    1 / Eq. 4.5 -- ✗ (maximisation)
    '''


    def sigLoss(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return -(s*s)/(s+b+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss

def significanceLossInvert(expectedSignal,expectedBkgd):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    Eq. 4.5 -- ✓
    '''

    def sigLossInvert(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)
        # signalWeight=1.#expectedSignal/K.sum(y_true)
        # bkgdWeight=1.#expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return (s+b)/(s*s+K.epsilon()) # Add the epsilon to avoid dividing by 0

    return sigLossInvert

def multiclass_aux(expectedSignal, expectedBkgd, systematic):

    def losses(y_true,y_pred):
        y_true_t = tf.transpose(y_true)
        y_pred_t = tf.transpose(y_pred)

        def get_true(i): return tf.transpose(y_true_t[i,:])
        def get_pred(i): return tf.transpose(y_pred_t[i,:])

        sample_weights = get_true(4)
        class_weights  = get_true(5)

        def crossentropy(i):
            y_true_ = get_true(i)
            y_pred_ = get_pred(i)#* class_weights
            # y_pred_ = get_pred(i) * sample_weights
            return keras.losses.categorical_crossentropy(y_true_, y_pred_)

        def significance(i):
            y_true_ = get_true(i)
            y_pred_ = get_pred(i)#* sample_weights

            signalWeight = expectedSignal / K.sum(y_true_)
            bkgdWeight = expectedBkgd / K.sum(1-y_true_)
            # signalWeight=1.#expectedSignal/K.sum(y_true_)
            # bkgdWeight=1.#expectedBkgd/K.sum(1-y_true_)

            s = signalWeight*K.sum(y_pred_*y_true_)
            b = bkgdWeight*K.sum(y_pred_*(1-y_true_))

            # Add the epsilon to avoid dividing by 0
            return (s+b)/(s*s+K.epsilon())

        def asimov(i):
            y_true_ = get_true(i)
            # y_pred_orig = get_pred(i)
            y_pred_ = get_pred(i)

            signalWeight = expectedSignal/K.sum(y_true_)
            bkgdWeight = expectedBkgd/K.sum(1-y_true_)
            # signalWeight = 1.#expectedSignal/K.sum(y_true_)
            # bkgdWeight = 1.#expectedBkgd/K.sum(1-y_true_)

            # dummy = K.print_tensor(K.min(y_pred_orig), "D1 =")
            # dummy2 = K.print_tensor(K.min(1-y_true_), "D2 =")
            # dummy3 = K.print_tensor(K.min(sample_weights), "D3 =")
            # dummy4 = K.print_tensor(K.min(y_pred_), "D4 =")
            s = signalWeight*K.sum(sample_weights*y_pred_*y_true_)
            # s = K.print_tensor(signalWeight*K.sum(y_pred_*y_true_) + 0*dummy + 0*dummy2 + 0*dummy3 + 0*dummy4, "s =")
            # b = K.print_tensor(bkgdWeight*K.sum(y_pred_*(1-y_true_)), "b =")
            b = bkgdWeight*K.sum(sample_weights*y_pred_*(1-y_true_))
            # b = K.print_tensor(b, "b =")
            b = tf.cond(b < 2, lambda: tf.constant(2.), lambda: b)
            sigB = systematic*b

            ln1_top = (s + b)*(b + sigB*sigB)
            ln1_bot = b*b + (s + b)*sigB*sigB
            ln1 = K.log(ln1_top / (ln1_bot + K.epsilon()) + K.epsilon())

            ln2 = K.log(1. + sigB*sigB*s / (b*(b + sigB*sigB) + K.epsilon()))

            return 1./(2*((s + b)*ln1 - b*b*ln2/(sigB*sigB + K.epsilon())) + K.epsilon()) #Add the epsilon to avoid dividing by 0

        return {
          'crossentropy': crossentropy,
          'significance': significance,
          'asimov'      : asimov,
          
          'sample_weights': sample_weights,
          'class_weights': class_weights,
        }

    return losses

def significanceLossInvert_m(expectedSignal,expectedBkgd):
    losses = multiclass_aux(expectedSignal, expectedBkgd, None)

    def sigLossInvert(y_true,y_pred):
        l = losses(y_true, y_pred)
        crossentropy = l['crossentropy']
        significance = l['significance']
        return crossentropy(0) \
             + crossentropy(1) \
             + crossentropy(2) \
             + significance(3)

    return sigLossInvert

def significanceLoss2Invert(expectedSignal,expectedBkgd):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    Eq. 4.5 (with b >> s) -- ✓
    '''

    def sigLoss2Invert(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return b/(s*s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss2Invert

def significanceLossInvertSqrt(expectedSignal,expectedBkgd):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    sqrt(Eq. 4.5) -- ✓
    '''

    def sigLossInvert(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))

        return K.sqrt(s+b)/(s+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLossInvert


def significanceFull(expectedSignal,expectedBkgd):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    sqrt(1 / Eq. 4.5) -- ✗ (maximisation)
    '''

    def significance(y_true,y_pred):
        #Discrete version

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(K.round(y_pred)*y_true)
        b = bkgdWeight*K.sum(K.round(y_pred)*(1-y_true))

        return s/K.sqrt(s+b+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return significance


def asimovSignificanceLoss(expectedSignal,expectedBkgd,systematic):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    -(Eq. 3.1)^2 -- ✓
    '''

    def asimovSigLoss(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))
        sigB=systematic*b

        ln1_top = (s + b)*(b + sigB*sigB)
        ln1_bot = b*b + (s + b)*sigB*sigB
        ln1 = K.log(ln1_top / (ln1_bot + K.epsilon()) + K.epsilon())

        ln2 = K.log(1. + sigB*sigB*s/(b*(b + sigB*sigB) + K.epsilon()))

        return 10000. -2.*((s + b)*ln1 - b*b*ln2/(sigB*sigB + K.epsilon())) #Add the epsilon to avoid dividing by 0

    return asimovSigLoss

def asimovSignificanceLossInvert(expectedSignal, expectedBkgd, systematic, debug=False):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    (1 / Eq. 3.1)^2 -- ✓
    '''

    def asimovSigLossInvert(y_true,y_pred):
        signalWeight = expectedSignal/K.sum(y_true)
        bkgdWeight = expectedBkgd/K.sum(1-y_true)
        # signalWeight = 1.#expectedSignal/K.sum(y_true)
        # bkgdWeight = 1.#expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(y_pred*y_true)
        b = bkgdWeight*K.sum(y_pred*(1-y_true))
        b = tf.cond(b < 2, lambda: tf.constant(2.), lambda: b)
        # b = K.print_tensor(b, "b =")
        sigB = systematic*b


        ln1_top = (s + b)*(b + sigB*sigB)
        ln1_bot = b*b + (s + b)*sigB*sigB
        ln1 = K.log(ln1_top / (ln1_bot + K.epsilon()) + K.epsilon())

        ln2 = K.log(1. + sigB*sigB*s / (b*(b + sigB*sigB) + K.epsilon()))

        if debug:
            s = K.print_tensor(s, message='s = ')
            b = K.print_tensor(b, message='b = ')
            sigB = K.print_tensor(sigB, message='sigB = ')
            ln1 = K.print_tensor(ln1, message='ln1 = ')
            ln2 = K.print_tensor(ln2, message='ln2 = ')

        loss = 1./(2*((s + b)*ln1 - b*b*ln2/(sigB*sigB + K.epsilon())) + K.epsilon()) #Add the epsilon to avoid dividing by 0
        if debug:
            loss = K.print_tensor(loss, message='loss = ')
        return loss

        # return 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB+K.epsilon())+K.epsilon())-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)+K.epsilon()))/(sigB*sigB+K.epsilon()))) #Add the epsilon to avoid dividing by 0

    return asimovSigLossInvert

def asimovSignificanceLossInvert_m(expectedSignal,
                                   expectedBkgd,
                                   systematic,
                                   debug=False):
    losses = multiclass_aux(expectedSignal, expectedBkgd, systematic)

    def asimovSigLossInvert(y_true,y_pred):
        l = losses(y_true, y_pred)
        crossentropy = l['crossentropy']
        asimov = l['asimov']
        sample_weights = l['sample_weights']
        class_weights = l['class_weights']

        def weight(score, weights):
            score_arr = (score * weights) \
                      / K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
            return K.mean(score_arr)

        loss_bg = weight(crossentropy(0), class_weights) \
                + weight(crossentropy(1), class_weights) \
                + weight(crossentropy(2), class_weights) \
                + weight(crossentropy(3), class_weights)
        loss_sig = asimov(3)
        
        if debug:
            loss_bg = K.print_tensor(loss_bg, "loss_bg = ")
            loss_sig = K.print_tensor(loss_sig, "loss_sig = ")
        return loss_sig + 1e-2*loss_bg

    return asimovSigLossInvert

def asimovSignificanceFull(expectedSignal,expectedBkgd,systematic):
    '''
    Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size.

    Eq. 3.1 -- ✗ (maximization)
    '''

    def asimovSignificance(y_true,y_pred):
        #Continuous version:

        signalWeight=expectedSignal/K.sum(y_true)
        bkgdWeight=expectedBkgd/K.sum(1-y_true)

        s = signalWeight*K.sum(K.round(y_pred)*y_true)
        b = bkgdWeight*K.sum(K.round(y_pred)*(1-y_true))
        sigB=systematic*b

        ln1_top = (s + b)*(b + sigB*sigB)
        ln1_bot = b*b + (s + b)*sigB*sigB
        ln1 = K.log(ln1_top / (ln1_bot + K.epsilon()) + K.epsilon())

        ln2 = K.log(1. + sigB*sigB*s / (b*(b + sigB*sigB) + K.epsilon()))

        return K.sqrt(2*((s + b)*ln1 - b*b*ln2/(sigB*sigB + K.epsilon()))) #Add the epsilon to avoid dividing by 0

    return asimovSignificance


def truePositive(y_true,y_pred):
    return K.sum(K.round(y_pred)*y_true) / (K.sum(y_true) + K.epsilon())

def falsePositive(y_true,y_pred):
    return K.sum(K.round(y_pred)*(1-y_true)) / (K.sum(1-y_true) + K.epsilon())
