#!/usr/bin/env python
########### author : Ashraf Kasem Mohamed ########## 
########### institute : DESY #######################
########### Email : ashraf.mohamed@desy.de #########
########### Date : May 2019 #######################
import sys,os, re, pprint
import re

import subprocess
import shutil
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import model_from_json
from root_pandas import read_root,to_root
import pandas as pd
import numpy as np

var_list = ['MET', 'MT', 'Jet2_pt','Jet1_pt', 'nLep', 'Lep_pt', 'Selected', 'nVeto', 'LT', 'HT', 'nBCleaned_TOTAL','nTop_Total_Combined', 'nJets30Clean', 'dPhi',"Lep_relIso","Lep_miniIso","iso_pt","iso_MT2","mGo", "mLSP"]
categoriesMultitarget = ['TTSemiLep','TTDiLep','WJets','Signal']
class eval(object):
    def __init__(self,infile,outdir,pathToModel,doBinary=False,do_multiClass = True,ClassList=None,mGo = 0 , mLSP = 0):
        self.infile        =  infile        
        self.outdir        =  outdir       
        self.pathToModel   =  pathToModel         
        self.doBinary      =  doBinary      
        self.do_multiClass =  do_multiClass 
        self.ClassList = ['TTSemiLep','TTDiLep','WJets','signal']
        self.mGo = mGo 
        self.mLSP = mLSP
        if not os.path.exists(self.outdir):
            os.makedirs(str(self.outdir))
            
    def load_model(self):
        '''Load a previously saved model (in h5 format)'''
        print (" Loading the model from ",self.pathToModel)
        # load json and create model
        json_file = open(self.pathToModel+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.pathToModel+'.h5')

    def varlist(self):
        var_file = open("1L_varList.txt",'r')
        L_varList = []
        for var in var_file :
            if var.startswith('#') : continue
            var = var.strip()
            L_varList.append(var)
        return L_varList
     
    def ev_score_toROOT(self,savepredicOnly=False):
        '''evaluate the score for the loaded modle'''
        L_varList = self.varlist()
        #get the model 
        self.load_model()
       
        df = read_root(self.infile ,'sf/t',columns=L_varList,flatten=['DLMS_ST','DLMS_HT','DLMS_dPhiLepW','DLMS_nJets30Clean'])
        print ('df loaded')
        print (" going to evalute the score from ",self.pathToModel)
        if not '_SMS_' in self.infile:
            df.loc[:,'mGo'] = float(self.mGo)
            df.loc[:,'mLSP'] = float(self.mLSP)
        print ('prediction will be made to mGo = ',self.mGo,' and mLSP =  ', self.mLSP )

        #print (df['mGo'].dtype)
        

        if self.do_multiClass : 
            self.model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
            prediction = self.model.predict_proba(df[var_list].values)
            if not savepredicOnly : 
                for mm, mult in enumerate(categoriesMultitarget) : 
                    df.loc[:,mult] = prediction[:,mm]
            else : 
                df = pd.DataFrame(prediction,columns=categoriesMultitarget)
        elif self.doBinary:
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            if not savepredicOnly : 
                df.loc[:,'DNN'] = self.model.predict(df[var_list])
            else :
                df = pd.DataFrame(self.model.predict(df[var_list]),columns=["DNN"])
        df.to_root(self.outdir+'/'+self.infile.split("/")[-1].replace('.root','_'+str(self.mGo)+'_'+str(self.mLSP)+'.root') ,key='sf/t')
        print ("out put fle is wrote to ",self.outdir+'/'+self.infile.split("/")[-1])

    def ev_score_toDF(self,savepredicOnly=False):
        '''evaluate the score for the loaded modle'''
        L_varList = self.varlist()
        #get the model 
        self.load_model()
        
        df = pd.read_csv(self.infile ,index_col=None)
        print ('df loaded')
        print (" going to evalute the score from ",self.pathToModel)
        if not '_SMS_' in self.infile:
            df.loc[:,'mGo'] = self.mGo
            df.loc[:,'mLSP'] = self.mLSP
        print('prediction will be made to mGo = ',self.mGo,' and mLSP =  ', self.mLSP )

        if self.do_multiClass : 
            self.model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
            prediction = self.model.predict_proba(df[var_list].values)
            if not savepredicOnly : 
                for mm, mult in enumerate(categoriesMultitarget) : 
                    df.loc[:,mult] = prediction[:,mm]
            else : 
                df = pd.DataFrame(prediction,columns=categoriesMultitarget)
        elif self.doBinary:
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            if not savepredicOnly : 
                df.loc[:,'DNN'] = self.model.predict(df[var_list])
            else : df = pd.DataFrame(self.model.predict(df[var_list]),columns=["DNN"])
            
        df.to_csv(self.outdir+'/'+self.infile.split("/")[-1].replace('.csv','_'+str(self.mGo)+'_'+str(self.mLSP)+'.csv'),index=None)
        print ("out put fle is wrote to ",self.outdir+'/'+self.infile.split("/")[-1])
