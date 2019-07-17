#!/usr/bin/env python
from preperData.splitDFs import splitDFs
from preperData.PrepData import PrepData
from MLClass.score import score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

## copied from A.Elwood https://github.com/aelwood/hepML/blob/master/MlFunctions/DnnFunctions.py
from MlFunctions.DnnFunctions import significanceLoss,significanceLossInvert,significanceLoss2Invert,significanceLossInvertSqrt,significanceFull,asimovSignificanceLoss,asimovSignificanceLossInvert,asimovSignificanceFull,truePositive,falsePositive

# if you want to use a pretrained model activate it and give the model path wthout any extension
loadmodel = False
pathToModel = './testing/model/1Lep_DNN_Multiclass'
append=''
##########################

# multiclass or binary 
MultiClass = True

if MultiClass : 
    class_names = ['TTSemiLep','TTDiLep','WJets','signal']
else : 
    class_names = ['signal','background']
##########################
# variables to be used in the training 
var_list = ['MET', 'MT', 'Jet2_pt','Jet1_pt', 'nLep', 'Lep_pt', 'Selected', 'nVeto', 'LT', 'HT', 'nBCleaned_TOTAL','nTop_Total_Combined', 'nJets30Clean', 'dPhi',"Lep_relIso","Lep_miniIso","iso_pt","iso_MT2","mGo", "mLSP"]

# variables to be used in while transfere DFs
VARS = ["MET","MT","Jet2_pt","Jet1_pt","nLep","Lep_pt","Selected","nVeto","LT","HT",
        "nBCleaned_TOTAL","nBJet","nTop_Total_Combined","nJets30Clean","dPhi","met_caloPt",
        "lheHTIncoming","genTau_grandmotherId","genTau_motherId","genLep_grandmotherId",
        "genLep_motherId","DiLep_Flag","semiLep_Flag","genWeight","sumOfWeights","btagSF",
        "puRatio","lepSF","nISRttweight","GenMET","Lep_relIso","Lep_miniIso","iso_pt","iso_MT2"]
##########################
# start preparing the data if it's not in place
Data = PrepData("/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/",'/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/csvs',VARS,skipexisting = False)
Data.saveCSV()
# preper the data and split them into testing sample + training sample
splitted = splitDFs(Data.df_all['sig'],Data.df_all['bkg'],do_multiClass = MultiClass,nSignal_Cla = 1,do_parametric = True,split_Sign_training = False)
splitted.prepare()
splitted.split(splitted.df_all['all_sig'],splitted.df_all['all_bkg'])
##########################
# init the modele 
scoreing = score('DNN','./testing',splitted.test_DF,splitted.train_DF,splitted.class_weights,var_list=var_list,do_multiClass = MultiClass,nSignal_Cla = 1,do_parametric = True,split_Sign_training = False,class_names=class_names)
# if continue pretrained model
if loadmodel : 
    append='_2nd'
    scoreing.load_model(pathToModel, loss='sparse_categorical_crossentropy') # mode will be save automatically
else : 
    # nClass will be ignored in binary classification tasks anywayes
    # loss = None will use the normal cross entropy change it if you want to whatever defined in MlFunctions/DnnFunctions.py
    scoreing.do_train(nclass =len(class_names),epochs=5,batch_size=1024,loss=None)
    #scoreing.load_model()
    scoreing.save_model(scoreing.model) # here we need to enforce saving it
##########################
# start the performance plottng 
# 1- the DNN score plots
from plotClass.pandasplot import pandasplot
import pandas as pd
train_s_df = pd.DataFrame(scoreing.dnn_score_train)
test_s_df = pd.DataFrame(scoreing.dnn_score_test)
full_test = pd.concat([scoreing.testDF,test_s_df],axis=1)
full_train = pd.concat([scoreing.trainDF,train_s_df],axis=1)
plott = pandasplot('./testing/',var_list)
plott.classifierPlot(full_test,full_train,norm=False,logY=True,append='',multiclass=MultiClass)
plott.var_plot(full_test,full_train,norm=False,logY=True,append='',multiclass=MultiClass,class_names=class_names)

# 2- the DNN loss and acc plotters 
scoreing.performance_plot(scoreing.history,scoreing.dnn_score_test,scoreing.dnn_score_train,append=append)

# 3- the DNN ROC plotters 
if MultiClass : 
    scoreing.rocCurve_multi(scoreing.dnn_score_test,label_binarize(splitted.test_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Test'+append,n_classes=4)
    scoreing.rocCurve_multi(scoreing.dnn_score_train,label_binarize(splitted.train_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Train'+append,n_classes=4)
else : 

    scoreing.rocCurve(scoreing.dnn_score_test,label_binarize(splitted.test_DF['isSignal'], classes=[0,1]),append='Binary_Test')
    scoreing.rocCurve(scoreing.dnn_score_train,label_binarize(splitted.train_DF['isSignal'], classes=[0,1]),append='Binary_Train')

# 4- the DNN confusion matrix plotters 
test_cm = confusion_matrix(splitted.test_DF["isSignal"],scoreing.dnn_score_test.argmax(axis=1))
train_cm = confusion_matrix(splitted.train_DF["isSignal"],scoreing.dnn_score_train.argmax(axis=1))

scoreing.plot_confusion_matrix(test_cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',append="test"+append)
scoreing.plot_confusion_matrix(train_cm, classes=class_names, normalize=True,
                               title='Normalized confusion matrix', append="train"+append)

# 5- the DNN correlation matrix plotters 
scoreing.heatMap(splitted.test_DF, append=append)
##########################
