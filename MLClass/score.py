
import numpy as np
import pandas as pd
import h5py
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
import concurrent.futures
from sklearn.metrics import log_loss, auc, roc_curve, accuracy_score, roc_auc_score
from keras.models import model_from_json
from xgboost import XGBClassifier
# baseline keras model
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adam,Nadam
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D,Dropout, BatchNormalization, Flatten,concatenate
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint ,EarlyStopping , ReduceLROnPlateau
from sklearn.model_selection import train_test_split
#from __future__ import division
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels
import itertools
import datetime
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

var_list = ['MET', 'MT', 'Jet2_pt','Jet1_pt' ,'nLep', 'Lep_pt', 'Selected', 'nVeto', 'LT', 'HT', 'nBCleaned_TOTAL',
            'nTop_Total_Combined', 'nJets30Clean', 'dPhi',"Lep_relIso",
             "Lep_miniIso","iso_pt","iso_MT2", 'mGo', 'mLSP']
class score(object):
    def __init__(self,score,outdir,testDF,trainDF,class_weights,do_binary= False,do_multiClass = True,nSignal_Cla = 1,do_parametric = True,split_Sign_training = False):
        self.score               = score                  
        self.outdir              = outdir                 
        self.testDF              = testDF                 
        self.trainDF             = trainDF                
        self.class_weights       = class_weights          
        self.do_binary           = do_binary        
        self.do_multiClass       = do_multiClass          
        self.nSignal_Cla         = nSignal_Cla            
        self.do_parametric       = do_parametric          
        self.split_Sign_training = split_Sign_training
        self.class_names         = ['TTSemiLep','TTDiLep','WJets','signal']

    # ## Define the model
    # We'll start with a dense (fully-connected) NN layer.
    # Our model will have a single fully-connected hidden layer with the same number of neurons as input variables. 
    # The weights are initialized using a small Gaussian random number. 
    # We will switch between linear and tanh activation functions for the hidden layer.
    # The output layer contains a single neuron in order to make predictions. 
    # It uses the sigmoid activation function in order to produce a probability output in the range of 0 to 1.
    # 
    # We are using the `binary_crossentropy` loss function during training, a standard loss function for binary classification problems. 
    # We will optimize the model with the Adam algorithm for stochastic gradient descent and we will collect accuracy metrics while the model is trained.

    # ## Run training 
    # Here, we run the training.

    # # Train XGB / DNN / etc.  

    # This is a nice isolated set of actions, so we will put them into a method right away
    def TrainXGB(self,train_DF,test_DF,var_list, n_estimators=150, max_depth=3, min_child_weight=1,seed=0):
        """ With training and testing DataFrame, and a list of variables with which to train and evaluate, produce the 
            score series for both the training and testing sets
        """
        
        
        # Create XGB object with the hyperparameters desired
        xgb = XGBClassifier(n_estimators=n_estimators,
                            max_depth=max_depth, 
                            min_child_weight=min_child_weight,
                            seed=seed)

        # Fit to the training set, making sure to include event weights
        xgb.fit(train_DF[var_list], # X
                train_DF["isSignal"], # yii
                sample_weight=train_DF["training_weight"], # weights
                )
        
                # Score the testing set
        Xg_score_test = xgb.predict(test_DF[var_list])#[:,0]  # predict_proba returns [prob_bkg, prob_sig] which have the property prob_bkg+prob_sig = 1 so we only need one. Chose signal-ness
        # Score the training set (for overtraining analysis)
        Xg_score_train = xgb.predict(train_DF[var_list])#[:,0] # predict_proba returns [prob_bkg, prob_sig] which have the property prob_bkg+prob_sig = 1 so we only need one. Chose signal-ness
        Xg_score_test = pd.Series(Xg_score_test, index=test_DF.index) 
        Xg_score_train = pd.Series(Xg_score_train, index=train_DF.index) 
        return Xg_score_train, Xg_score_test


    def TrainDNN(self,train_DF,test_DF,var_list,multi=False,nclass = 4,epochs=200,batch_size=1024,useDropOut = False,class_weights=None):
        """ With training and testing DataFrame, and a list of variables with which to train and evaluate, produce the 
            score series for both the training and testing sets
        """
        NDIM = len(var_list)
        DNN = Sequential()
        DNN.add(Dense(256, input_dim=NDIM, kernel_initializer='uniform', activation='relu'))
        if useDropOut : 
            DNN.add(Dropout(0.01))
        DNN.add(Dense(256, kernel_initializer='uniform', activation='relu'))
        if useDropOut : 
            DNN.add(Dropout(0.01))
        DNN.add(Dense(256, kernel_initializer='uniform', activation='relu'))
        # Compile model
        if multi == False :
            if useDropOut : 
                DNN.add(Dropout(0.01))
            DNN.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
            DNN.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer=Adam(lr=0.0001))
        elif multi == True :
            if useDropOut : 
                DNN.add(Dropout(0.01))
            DNN.add(Dense(nclass, kernel_initializer='uniform', activation='softmax'))
            DNN.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'], optimizer=Adam(lr=0.001))

        DNN.summary()
        # Train DNN classifier
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            # model checkpoint callback
            # this saves our model architecture + parameters into dense_model.h5
        model_checkpoint = ModelCheckpoint('dense_model.h5', monitor='val_loss', 
                            verbose=0, save_best_only=True, 
                            save_weights_only=False, mode='auto', 
                            period=1)

        history = DNN.fit(train_DF[var_list].values, 
                                train_DF["isSignal"].values,
                                epochs=epochs,
                                batch_size=batch_size, 
                                class_weight = class_weights,#train_DF["Finalweight"].values,#*train_DF["training_weight"],
                                verbose=1, # switch to 1 for more verbosity 
                                callbacks=[early_stopping, model_checkpoint], 
                                validation_split=0.25)


        dnn_score_test = DNN.predict(test_DF[var_list])
        dnn_score_train = DNN.predict(train_DF[var_list])
        ## better also to return the model it self 
        return dnn_score_test, dnn_score_train, history , DNN
    
    def do_train(self):
        "do the actual training"
        if self.score == 'DNN' : 
            print('let\'s do DNN training')
            ## Run the method we have just defined
            print (self.trainDF.groupby(['isSignal']).size())
            self.dnn_score_test, self.dnn_score_train,self.history ,self.model = self.TrainDNN(self.trainDF, 
                                                    self.testDF,
                                                    var_list,
                                                    multi=self.do_multiClass,
                                                    nclass =4,
                                                    epochs=10,
                                                    batch_size=1024,
                                                    useDropOut = True)
        elif self.score == 'XGB' : 
            #Run XGB
            print('let\'s do XGBoost training')
            self.Xg_score_train, self.Xg_score_test = self.TrainEval(self.trainDF, 
                                              self.testDF,
                                              var_list,
                                              n_estimators=200, 
                                              max_depth=3,
                                              min_child_weight=1,
                                              seed=0,
                                              cla = 'xgb')
            #PlotROCTemplate(test_DF, train_DF, Xg_score_test, Xg_score_train)
        
    def save_model(self,model_toSave):
        '''Save the model'''
        output=os.path.join(self.outdir,'model')
        if not os.path.exists(output): os.makedirs(output)
        # serialize model to JSON
        model_json = model_toSave.to_json()
        with open(output+"/1Lep_DNN_Multiclass.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model_toSave.save_weights(output+"/1Lep_DNN_Multiclass.h5")
        print("Saved model to disk")
        json_file.close()
        
    def load_model(self,pathToModel):
        '''Load a previously saved model (in h5 format)'''
        # load json and create model
        json_file = open(pathToModel+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(pathToModel+'.h5')

    def performance_plot(self,history,dnn_score_test,dnn_score_train):
        # Make an example plot with two subplots...
        """
        overtraining test
        """
        print (' plotting the peformace ')
        # Extract number of run epochs from the training history
        epochs = range(1, len(history.history["loss"])+1)
        plt.figure(figsize=(9, 4))
        #fig = plt.figure(figsize=(4, 4))
        plt.subplot(1, 2, 1)
        # Extract loss on training and validation dataset and plot them together
        plt.plot(epochs, history.history["loss"], "o-", label="Training")
        plt.plot(epochs, history.history["val_loss"], "o-", label="Validation")
        plt.xlabel("Epochs"), plt.ylabel("Loss")
        #plt.yscale("log")
        #plt.xlim(0,40)
        #plt.ylim(.3,0.9)
        plt.grid()
        plt.legend();
        ax = plt.gca()
        # recompute the ax.dataLim
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()
        plt.subplot(1, 2, 2)
        #fig = plt.figure(figsize=(4, 4))
        # Extract loss on training and validation dataset and plot them together
        plt.plot(epochs, history.history["acc"], "o-", label="Training")
        plt.plot(epochs, history.history["val_acc"], "o-", label="Validation")
        plt.xlabel("Epochs"), plt.ylabel("Accuracy")
        #plt.yscale("log")
        #plt.ylim(0.5,0.95)
        plt.grid()
        plt.legend(loc="best");
        ax = plt.gca()
        # recompute the ax.dataLim
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()
        plt.subplots_adjust(bottom=0.15, wspace=0.30)
        outputplot=os.path.join(self.outdir,'plots')
        if not os.path.exists(outputplot): os.makedirs(outputplot)
        plt.savefig(outputplot+'/performance.pdf')
        plt.clf()
        #plt.show()
        # Draw the Roc curves for testing and training samples

    def rocCurve(self,y_preds,y_test=None,append=''):
        '''Compute the ROC curves, can either pass the predictions and the truth set or 
        pass a dictionary that contains one value 'truth' of the truth set and the other 
        predictions labeled as you want'''
        print (' plotting the ROC for binary score ')
        # Compute ROC curve and area under the curve
        if not isinstance(y_preds,dict):
            assert not y_test is None,'Need to include testing set if not passing dict'
            y_preds={'ROC':y_preds}
        else:
            y_test=y_preds['truth']

        for name,y_pred in y_preds.iteritems():
            if name=='truth': continue
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=1, label=name+' (area = %0.2f)'%(roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.grid()
        outputplot=os.path.join(self.outdir,'plots')
        if not os.path.exists(outputplot): os.makedirs(outputplot)
        plt.savefig(os.path.join(outputplot,'rocCurve'+append+'.pdf'))
        plt.clf()


    def rocCurve_multi(self,y_preds,y_test=None,append='',n_classes=4):    # Compute ROC curve and ROC area for each class
        print (' plotting the ROC for multiClass score ')
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)
        lw = 2
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','darkolivegreen','red','brown'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right" , prop={'size': 10})
        plt.grid()
        outputplot=os.path.join(self.outdir,'plots')
        if not os.path.exists(outputplot): os.makedirs(outputplot)
        plt.savefig(os.path.join(outputplot,'rocCurve'+append+'.pdf'))
        plt.clf()
        #plt.show()

    def plot_confusion_matrix(self,cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues,append=''):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        print (' plotting the confusion matrix for multiClass score '+append)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix "+append)
        else:
            print('Confusion matrix, without normalization '+append)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.rcParams["figure.figsize"] = [20,9]
        outputplot=os.path.join(self.outdir,'plots')
        if not os.path.exists(outputplot): os.makedirs(outputplot)
        plt.savefig(outputplot+'/confusion_matrix_'+append+'.pdf')
        plt.clf()
        #plt.show()

    def heatMap(self,DFrame,var_list=var_list):
        import seaborn
        outputplot=os.path.join(self.outdir,'plots')
        if not os.path.exists(outputplot): os.makedirs(outputplot)
        
        for multitarget in range(0,len(self.class_names)) :
            print ("multitarget ="+str(multitarget))
            corr_mat = DFrame.loc[(DFrame["isSignal"]==multitarget), var_list].astype(float).corr() #
            fig, ax = plt.subplots(figsize=(20, 12)) 
            Hmap = seaborn.heatmap(corr_mat, square=True, ax=ax, vmin=-1., vmax=1.,annot=True)
            Hmap.figure.savefig(outputplot+'/Class_'+str(self.class_names[multitarget])+'_hmx.pdf', transparent=True, bbox_inches='tight')
    
#    def compareTrainTest(self,clf, X_train, y_train, X_test, y_test, bins=30,append=''):
#        '''Compares the decision function for the train and test BDT'''
#        decisions = []
#        if self.do_multiClass : 
#            for i in range(0,len(self.class_names)) :
#                decisions += clf[i]
#        elif self.do_binary_first : 
#            decisions += [clf[0],clf[1]]
#
#        low = min(np.min(d) for d in decisions)
#        high = max(np.max(d) for d in decisions)
#        low_high = (low,high)
#        
#        for i, d in decisions : 
#            plt.hist(d, color='r', alpha=0.5, range=low_high, bins=bins,
#                histtype='stepfilled', normed=True,
#                label='S (train)')
#        plt.hist(decisions[1],
#                color='b', alpha=0.5, range=low_high, bins=bins,
#                histtype='stepfilled', normed=True,
#                label='B (train)')
#
#        hist, bins = np.histogram(decisions[2],
#                                bins=bins, range=low_high, normed=True)
#        scale = len(decisions[2]) / sum(hist)
#        err = np.sqrt(hist * scale) / scale
#        
#        width = (bins[1] - bins[0])
#        center = (bins[:-1] + bins[1:]) / 2
#        plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
#        
#        hist, bins = np.histogram(decisions[3],
#                                bins=bins, range=low_high, normed=True)
#        scale = len(decisions[3]) / sum(hist)
#        err = np.sqrt(hist * scale) / scale
#
#        plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
#
#        plt.xlabel("Classifier output")
#        plt.ylabel("Arbitrary units")
#        plt.yscale('log')
#        plt.legend(loc='best')
#        outputplot=os.path.join(self.outdir,'plots')
#        if not os.path.exists(outputplot): os.makedirs(outputplot)
#        plt.savefig(os.path.join(outputplot, 'compareTrainTest'+append+'.pdf'))
#        plt.clf()
