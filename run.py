#!/usr/bin/env python
from preperData.splitDFs import splitDFs
from preperData.PrepData import PrepData
from MLClass.score import score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

Data = PrepData("/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/",'/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/csvs')
Data.saveCSV()
splitted = splitDFs(Data.df_all['sig'],Data.df_all['bkg'])
splitted.prepare()
splitted.split(splitted.df_all['all_sig'],splitted.df_all['all_bkg'])
scoreing = score('DNN','./testing',splitted.test_DF,splitted.train_DF,splitted.class_weights)
scoreing.do_train()
scoreing.save_model(scoreing.model)
scoreing.performance_plot(scoreing.history,scoreing.dnn_score_test,scoreing.dnn_score_train)
scoreing.rocCurve_multi(scoreing.dnn_score_test,label_binarize(splitted.test_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Test',n_classes=4)
scoreing.rocCurve_multi(scoreing.dnn_score_train,label_binarize(splitted.train_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Train',n_classes=4)


test_cm = confusion_matrix(splitted.test_DF["isSignal"],scoreing.dnn_score_test.argmax(axis=1))
train_cm = confusion_matrix(splitted.train_DF["isSignal"],scoreing.dnn_score_train.argmax(axis=1))

class_names = ['TTSemiLep','TTDiLep','WJets','signal']#'Sig1500','Sig1700','Sig1900','Sig2100']#,'Sig1000','Sig1200','Sig1400']#,'Sig_1','Sig_2']
# Plot normalized confusion matrix
scoreing.plot_confusion_matrix(test_cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',append="test")
scoreing.plot_confusion_matrix(train_cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',append="train")
scoreing.heatMap(splitted.test_DF)

scoreing.compareTrainTest(clf, X_train, y_train, X_test, y_test, output, bins=30,append='')