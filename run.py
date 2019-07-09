#!/usr/bin/env python
from preperData.splitDFs import splitDFs
from preperData.PrepData import PrepData
from MLClass.score import score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

loadmodel = False
pathToModel = './testing/model/1Lep_DNN_Multiclass'
append=''


var_list = ['MET', 'MT', 'Jet2_pt','Jet1_pt', 'nLep', 'Lep_pt', 'Selected', 'nVeto', 'LT', 'HT', 'nBCleaned_TOTAL','nTop_Total_Combined', 'nJets30Clean', 'dPhi',"Lep_relIso","Lep_miniIso","iso_pt","iso_MT2","mGo", "mLSP"]


Data = PrepData("/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/",'/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/csvs')

Data.saveCSV()
splitted = splitDFs(Data.df_all['sig'],Data.df_all['bkg'])
splitted.prepare()
splitted.split(splitted.df_all['all_sig'],splitted.df_all['all_bkg'])
scoreing = score('DNN','./testing',splitted.test_DF,splitted.train_DF,splitted.class_weights,var_list=var_list)
if loadmodel : 
    append='_2nd'
    scoreing.load_model(pathToModel, loss='sparse_categorical_crossentropy') # mode will be save automatically
else : 
    scoreing.do_train(nclass =4,epochs=10,batch_size=1024)
    #scoreing.load_model()
    scoreing.save_model(scoreing.model) # here we need to enforce saving it

scoreing.performance_plot(scoreing.history,scoreing.dnn_score_test,scoreing.dnn_score_train,append=append)
scoreing.rocCurve_multi(scoreing.dnn_score_test,label_binarize(splitted.test_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Test'+append,n_classes=4)
scoreing.rocCurve_multi(scoreing.dnn_score_train,label_binarize(splitted.train_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Train'+append,n_classes=4)


test_cm = confusion_matrix(splitted.test_DF["isSignal"],scoreing.dnn_score_test.argmax(axis=1))
train_cm = confusion_matrix(splitted.train_DF["isSignal"],scoreing.dnn_score_train.argmax(axis=1))

class_names = ['TTSemiLep','TTDiLep','WJets','signal']#'Sig1500','Sig1700','Sig1900','Sig2100']#,'Sig1000','Sig1200','Sig1400']#,'Sig_1','Sig_2']
# Plot normalized confusion matrix
scoreing.plot_confusion_matrix(test_cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',append="test"+append)
scoreing.plot_confusion_matrix(train_cm, classes=class_names, normalize=True,
                               title='Normalized confusion matrix', append="train"+append)
scoreing.heatMap(splitted.test_DF, append=append)


#scoreing.compareTrainTest(clf, X_train, y_train, X_test, y_test, output, bins=30,append='')
