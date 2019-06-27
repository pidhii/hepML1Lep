import numpy as np
import pandas as pd
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import os
Mass_points = [[1900,1000]]#,[2200,100],[2200,800],[1900,800],[1900,100],[1500,1000],[1500,1200],[1700,1200],[1600,1100],[1800,1300]]
signal_Cla = [[[1600,1100],[1800,1300],[1500,1000],[1500,1200],[1700,1200]],[[1900,100],[2200,100],[2200,800],[1900,800],[1900,1000]]]
do_hyperOpt = False
to_drop = ['lheHTIncoming', 'genTau_grandmotherId', 'genTau_motherId', 'genLep_grandmotherId',
               'genLep_motherId', 'DiLep_Flag', 'semiLep_Flag', 'GenMET',  'filename']

var_list = ['MET', 'MT', 'Jet2_pt','Jet1_pt' ,'nLep', 'Lep_pt', 'Selected', 'nVeto', 'LT', 'HT', 'nBCleaned_TOTAL',
    'nTop_Total_Combined', 'nJets30Clean', 'dPhi',"Lep_relIso",
    "Lep_miniIso","iso_pt","iso_MT2"]

class splitDFs(object):
    def __init__(self,signalDF, bkgDF,do_multiClass = True,nSignal_Cla = 1,do_parametric = True,split_Sign_training = False):
        self.signalDF = signalDF
        self.bkgDF = bkgDF
        #self.do_binary_first = do_binary_first
        self.do_multiClass = do_multiClass
        self.nSignal_Cla =nSignal_Cla
        self.do_parametric = do_parametric
        self.split_Sign_training = split_Sign_training
    # function to get the index of each class of background
    def classidxs(self):
        self.SemiLep_TT_index   = self.bkgDF[self.bkgDF['filename'].str.contains('TTJets_SingleLeptonFrom')].index
        self.DiLep_TT_index     = self.bkgDF[self.bkgDF['filename'].str.contains('TTJets_DiLepton')].index
        #QCD_index        = self.bkgDF[self.bkgDF['filename'].str.contains('QCD')].index
        self.WJets_others_index = self.bkgDF[~ self.bkgDF['filename'].str.contains('TTJets')].index

        print (self.signalDF.groupby(['mGo','mLSP']).size())

        ## this is very usful when you need to sample specific class to match with other class (overSample Signal to backgound for example)        
    from sklearn.utils import shuffle
    def _overbalance(self,train_s,train_bkg):
        """
        Return Oversampled dataset
        """
        count_s = len(train_s.index)
        count_bkg = len(train_bkg.index)
        # Divide by class
        df_class_0 = train_bkg
        df_class_1 = train_s
        df_class_1_over = df_class_1.sample(count_bkg, replace=True)
        df_class_1_over = shuffle(df_class_1_over)
        return df_class_1_over

    ## this is very usful when you need to sample background class to preper it for the parametric training
    def _overbalance_bkg(self,signals_df_list,bkg_df):
        new_bkg_train = pd.DataFrame()
        bkg_df = bkg_df.copy()
        for ns in signals_df_list : 
            bkg_df.loc[:,'mGo'] = np.random.choice(list(ns['mGo']), len(bkg_df))
            bkg_df.loc[:,'mLSP'] = np.random.choice(list(ns['mLSP']), len(bkg_df))
            new_bkg_train = pd.concat([new_bkg_train, bkg_df], ignore_index=True)
        return new_bkg_train

        
    def sigidxs(self):
        self.list_of_mass_idxs = []
        self.signal_list_names = [] 
        for massP in Mass_points:
            print ('mass chosen is [mGo,mLSP] == : ', massP)
            vars()['Sig_index_mGo_'+str(massP[0])+'_mLSP_'+str(massP[1])] = self.signalDF.index[(self.signalDF['mGo'] == massP[0]) & (self.signalDF['mLSP'] == massP[1])]
            self.list_of_mass_idxs.append(vars()['Sig_index_mGo_'+str(massP[0])+'_mLSP_'+str(massP[1])])
            self.signal_list_names.append('Sig_'+str(massP[0])+'_'+str(massP[1]))

    def prepare(self):
        self.classidxs()
        self.sigidxs()
        self.df_all = {}
        self.df_all['all_sig'] = pd.DataFrame()

        if self.nSignal_Cla > 1 and self.do_multiClass: 
            self.bkgDF.loc[self.SemiLep_TT_index,'isSignal'] = 0 #pd.Series(np.zeros(self.bkgDF.shape[0]), index=self.bkgDF.index)
            self.bkgDF.loc[self.DiLep_TT_index,'isSignal'] = 1
            self.bkgDF.loc[self.WJets_others_index,'isSignal'] = 2
            for num ,idxs in enumerate(self.list_of_mass_idxs) : 
                    self.df_all[self.signal_list_names[num]] = self.signalDF.loc[idxs ,:]
                    for j ,i in  enumerate(signal_Cla[0]) : 
                        if str(i[0]) in self.signal_list_names[num] and str(i[1]) in self.signal_list_names[num] : 
                            print (i , j ,self.signal_list_names[num])
                            self.signalDF.loc[idxs,'isSignal'] = 3
                    for j ,i in  enumerate(signal_Cla[1]) : 
                        if str(i[0]) in self.signal_list_names[num] and str(i[1]) in self.signal_list_names[num] : 
                            print (i , j ,self.signal_list_names[num])
                            self.signalDF.loc[idxs,'isSignal'] = 4
            self.df_all['all_sig'] = self.signalDF.copy()
            self.df_all['all_sig'] = self.df_all['all_sig'].dropna()
            if self.do_parametric : 
                signal_list_dfs = [] 
                for name in  self.signal_list_names : 
                    signal_list_dfs.append(self.df_all[name])
                    #print signal_list_dfs
                self.df_all['all_bkg'] = self._overbalance_bkg(signal_list_dfs,self.bkgDF)
            else : self.df_all['all_bkg'] = self.bkgDF.copy()
            ## free up the memeory from all other dfs 
            bkgdf =  self.df_all['all_bkg'].copy()
            sigdf =  self.df_all['all_sig'].copy()
            del self.df_all
            self.df_all = {}
            self.df_all['all_bkg'] = bkgdf.copy()
            self.df_all['all_sig'] = sigdf.copy()
            del bkgdf
            del sigdf
    
        elif self.do_multiClass and not self.split_Sign_training : 
            self.bkgDF.loc[self.SemiLep_TT_index,'isSignal'] = 0 #pd.Series(np.zeros(self.bkgDF.shape[0]), index=self.bkgDF.index)
            self.bkgDF.loc[self.DiLep_TT_index,'isSignal'] = 1
            self.bkgDF.loc[self.WJets_others_index,'isSignal'] = 2
            
            for num ,idxs in enumerate(self.list_of_mass_idxs) :
                self.df_all[self.signal_list_names[num]] = self.signalDF.loc[idxs ,:]
                self.df_all[self.signal_list_names[num]].loc[:,'isSignal'] = 3
                ## for the last training over all the samples (the multiClass trainig)
                self.df_all['all_sig'] = pd.concat([self.df_all['all_sig'],self.df_all[self.signal_list_names[num]]])
                #del self.df_all[self.signal_list_names[num]]
            if self.do_parametric : 
                signal_list_dfs = [] 
                for name in  self.signal_list_names : 
                    signal_list_dfs.append(self.df_all[name])
                #print signal_list_dfs
                self.df_all['all_bkg'] = self._overbalance_bkg(signal_list_dfs,self.bkgDF)
            else : self.df_all['all_bkg'] = self.bkgDF.copy()
            ## free up the memeory from all other dfs 
            bkgdf =  self.df_all['all_bkg'].copy()
            sigdf =  self.df_all['all_sig'].copy()
            del self.df_all
            self.df_all = {}
            self.df_all['all_bkg'] = bkgdf.copy()
            self.df_all['all_sig'] = sigdf.copy()
            del bkgdf
            del sigdf
    
        elif self.split_Sign_training and self.do_multiClass: 
            self.bkgDF.loc[self.SemiLep_TT_index,'isSignal'] = 0 #pd.Series(np.zeros(self.bkgDF.shape[0]), index=self.bkgDF.index)
            self.bkgDF.loc[self.DiLep_TT_index,'isSignal'] = 1
            self.bkgDF.loc[self.WJets_others_index,'isSignal'] = 2
            self.df_all['sig_1'] = self.signalDF.copy()
            self.df_all['sig_2'] = self.signalDF.copy()
            for num ,idxs in enumerate(self.list_of_mass_idxs) : 
                    self.df_all[self.signal_list_names[num]] = self.signalDF.loc[idxs ,:]
                    for j ,i in  enumerate(signal_Cla[0]) : 
                        if str(i[0]) in self.signal_list_names[num] and str(i[1]) in self.signal_list_names[num] : 
                            print (i , j ,self.signal_list_names[num])
                            self.df_all['sig_1'].loc[idxs,'isSignal'] = 3
                    for j ,i in  enumerate(signal_Cla[1]) : 
                        if str(i[0]) in self.signal_list_names[num] and str(i[1]) in self.signal_list_names[num] : 
                            print (i , j ,self.signal_list_names[num])
                            self.df_all['sig_2'].loc[idxs,'isSignal'] = 3
            self.df_all['all_sig_1'] = self.df_all['sig_1'].copy()
            self.df_all['all_sig_2'] = self.df_all['sig_2'].copy()
            self.df_all['all_sig_1'] = self.df_all['all_sig_1'].dropna()
            self.df_all['all_sig_2'] = self.df_all['all_sig_2'].dropna()
            if self.do_parametric : 
                signal_list_dfs_1 = [] 
                signal_list_dfs_2 = [] 
                for name in  self.signal_list_names :
                    print (signal_Cla[0])
                    for scla in signal_Cla[0] :
                        if name == 'Sig_'+str(scla[0])+'_'+str(scla[1]) : 
                                signal_list_dfs_1.append(self.df_all[name])
                    print (signal_Cla[1])
                    for scla in signal_Cla[1] :
                        if name == 'Sig_'+str(scla[0])+'_'+str(scla[1]) : 
                                signal_list_dfs_2.append(self.df_all[name])
                    
                    #print signal_list_dfs
                self.df_all['all_bkg_1'] = self._overbalance_bkg(signal_list_dfs_1,self.bkgDF)
                self.df_all['all_bkg_2'] = self._overbalance_bkg(signal_list_dfs_2,self.bkgDF)
            else : self.df_all['all_bkg'] = self.bkgDF.copy()

        elif not self.do_multiClass : 
            for num ,idxs in enumerate(self.list_of_mass_idxs) : 
                self.df_all[self.signal_list_names[num]] = self.signalDF.loc[idxs ,:]
                self.df_all[self.signal_list_names[num]].loc[:,'isSignal'] = 1
                ## for the last training over all the samples (the multiClass trainig)
                self.df_all['all_sig'] = pd.concat([self.df_all['all_sig'],self.df_all[self.signal_list_names[num]]])
                #del self.df_all[self.signal_list_names[num]]

            self.df_all['all_sig'] = self.df_all['all_sig'].reset_index()    
            self.df_all['all_bkg'].loc[self.SemiLep_TT_index,'isSignal'] = 0
            self.df_all['all_bkg'].loc[self.DiLep_TT_index,'isSignal'] = 0
            self.df_all['all_bkg'].loc[self.WJets_others_index,'isSignal'] = 0
            if self.do_parametric : 
                signal_list_dfs = [] 
                for name in  self.signal_list_names : 
                    signal_list_dfs.append(self.df_all[name])
                #print signal_list_dfs
                self.df_all['all_bkg'] = self._overbalance_bkg(signal_list_dfs,self.bkgDF)
            else : self.df_all['all_bkg'] = self.bkgDF.copy()
            ## free up the memeory from all other dfs 
            bkgdf =  self.df_all['all_bkg'].copy()
            sigdf =  self.df_all['all_sig'].copy()
            del self.df_all
            self.df_all = {}
            self.df_all['all_bkg'] = bkgdf.copy()
            self.df_all['all_sig'] = sigdf.copy()
            del bkgdf
            del sigdf

    def split(self,sigdfnew,bkgdfnew,train_size=0.6, test_size=0.4, shuffle=True, random_state=0) :
        print ('now splitting the samples with the options : ','train_size = ', train_size, 'test_size = ',test_size, 'shuffle = ',shuffle, 'random_state = ',random_state)
        # write df1 content in file.csv
        
        _df_all = pd.concat([sigdfnew,bkgdfnew])
        del sigdfnew, bkgdfnew
        _df_all_tr = _df_all.drop(to_drop,axis=1)
        self.train_DF, self.test_DF = train_test_split(_df_all_tr, train_size=train_size, test_size=test_size, shuffle=shuffle, random_state=random_state)        
        self.train_DF = self.train_DF.reset_index(drop=True)
        self.test_DF  = self.test_DF.reset_index(drop=True)

        self.class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(self.train_DF['isSignal']),
                                                self.train_DF['isSignal'])
        # write the testDF and trainDF in case you want to save time (not too much)
        
    
        #if self.do_binary_first : 
        #    for num ,idxs in enumerate(self.list_of_mass_idxs) : 
        #            self.df_all[self.signal_list_names[num]] = self.signalDF.loc[idxs ,:]
        #            self.df_all[self.signal_list_names[num]].loc[:,'isSignal'] = 1
        #            ## for the last training over all the samples (the multiClass trainig)
        #            self.df_all['all_sig'] = pd.concat([self.df_all['all_sig'],self.df_all[self.signal_list_names[num]]])
        #    self.df_all['all_sig'].loc[:,'isSignal'] = 3 
        #    self.df_all['all_sig'] = self.df_all['all_sig'].reset_index()    
        #    # for binary classifiers first
        #    self.bkgDF.loc[self.SemiLep_TT_index,'isSignal'] = 0 #pd.Series(np.zeros(self.bkgDF.shape[0]), index=self.bkgDF.index)
        #    self.bkgDF.loc[self.DiLep_TT_index,'isSignal'] = 0
        #    self.bkgDF.loc[self.WJets_others_index,'isSignal'] = 0
        #    # save it unchanged for binary classification iterations
        #    # combine the background for the last step 
        #    self.df_all['all_bkg'] = self.bkgDF.copy()
        #    # locate the class number
        #    self.df_all['all_bkg'].loc[self.SemiLep_TT_index,'isSignal'] = 0
        #    self.df_all['all_bkg'].loc[self.DiLep_TT_index,'isSignal'] = 1
        #    self.df_all['all_bkg'].loc[self.WJets_others_index,'isSignal'] = 2
        #    self.df_all['all_bkg'] = self.df_all['all_bkg'].reset_index()
        #    ## signal list for binary classifications
        #    #print signal_list_dfs
        #    if self.do_parametric :
        #        signal_list_dfs = [] 
        #        for name in  self.signal_list_names : 
        #            signal_list_dfs.append(self.df_all[name])
        #        self.df_all['all_bkg'] = self._overbalance_bkg(signal_list_dfs,self.df_all['all_bkg'])
        #    else : self.df_all['all_bkg'] = self.bkgDF.copy()
        #    ## free up the memeory from all other dfs 
        #    bkgdf =  self.df_all['all_bkg'].copy()
        #    sigdf =  self.df_all['all_sig'].copy()
        #    del self.df_all
        #    self.df_all = {}
        #    self.df_all['all_bkg'] = bkgdf.copy()
        #    self.df_all['all_sig'] = sigdf.copy()
        #    del bkgdf
        #    del sigdf