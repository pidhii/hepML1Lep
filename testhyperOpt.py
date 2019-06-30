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

from MLClass.hyperOpt import hyperOpt

hyper = hyperOpt('./testing',splitted.train_DF,splitted.class_weights)
hyper.do_gridsearch(useDropOut=True,multi=True)
