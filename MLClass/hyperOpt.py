#!/usr/bin/env python
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam

var_list = ['MET', 'MT', 'Jet2_pt','Jet1_pt' ,'nLep', 'Lep_pt', 'Selected', 'nVeto', 'LT', 'HT', 'nBCleaned_TOTAL',
            'nTop_Total_Combined', 'nJets30Clean', 'dPhi',"Lep_relIso",
             "Lep_miniIso","iso_pt","iso_MT2", 'mGo', 'mLSP']

class hyperOpt(object): 
    def __init__(self, outdir, trainDF, class_weights):
        self.outdir              = outdir                 
        self.trainDF             = trainDF                
        self.class_weights       = class_weights          
        self.defaultParams = {} 
        self.class_names         = ['TTSemiLep','TTDiLep','WJets','signal']

    def create_model(self,dropout_rate=0,learn_rate=0.01,loss=None,useDropOut=True,multi=True):
        # create model
        NDIM = len(var_list)
        model = Sequential()
        model.add(Dense(256, input_dim=NDIM, kernel_initializer='uniform', activation='relu'))
        if useDropOut : 
            model.add(Dropout(dropout_rate))
        model.add(Dense(256, kernel_initializer='uniform', activation='relu'))
        if useDropOut : 
            model.add(Dropout(dropout_rate))
        model.add(Dense(256, kernel_initializer='uniform', activation='relu'))
        # Compile model
        if multi == False :
            if useDropOut : 
                model.add(Dropout(dropout_rate))
            model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
            # if you want to change the loss function in the first stage
            if loss is not None : 
                model.compile(loss=loss,metrics=['accuracy'], optimizer=Adam(lr=learn_rate))
            else : 
                model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer=Adam(lr=learn_rate))
        elif multi == True :
            if useDropOut : 
                model.add(Dropout(dropout_rate))
            model.add(Dense(len(self.class_names),
                            kernel_initializer='uniform', activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'], optimizer=Adam(lr=learn_rate))
        return model
    def do_gridsearch(self,useDropOut = False,multi=False):
        '''Implementation of the sklearn grid search for hyper parameter tuning, 
        making use of kfolds cross validation.
        Pass a dictionary of lists of parameters to test on. Choose number of cores
        to run on with n_jobs, -1 is all of them'''
        #Reference:https://github.com/aelwood/hepML/blob/master/MlClasses/Dnn.py#L159 which is taken from 
        #https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)

        model = KerasClassifier(build_fn=self.create_model, verbose=0)
        # define the grid search parameters
        batch_size = [256,512,1024,2048]
        epochs = [10, 50, 100]
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        #momentum = [0.0, 0.2, 0.4]
        dropout_rate = [0.0,0.01,0.1, 0.2, 0.3, 0.4]
        X = self.trainDF[var_list].values
        Y = self.trainDF['isSignal'].values
        w = self.class_weights
        param_grid = dict(batch_size=batch_size, epochs=epochs,
                          learn_rate=learn_rate, dropout_rate=dropout_rate)  # momentum=momentum ,
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        self.grid_result = grid.fit(X, Y )

        #Save the results
        if not os.path.exists(self.outdir): os.makedirs(self.outdir)
        outFile = open(os.path.join(self.outdir, 'gridSearchResults.txt'), 'w')
        outFile.write("Best: %f using %s \n\n" % (self.grid_result.best_score_, self.grid_result.best_params_))
        means = self.grid_result.cv_results_['mean_test_score']
        stds = self.grid_result.cv_results_['std_test_score']
        params = self.grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            outFile.write("%f (%f) with: %r\n" % (mean, stdev, param))
        outFile.close()
