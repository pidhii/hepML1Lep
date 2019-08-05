import uproot
import numpy as np
import pandas as pd
import h5py
import os
seed = 7
np.random.seed(seed)
import concurrent.futures

import itertools
import datetime
import shutil

treename = 'sf/t'
filename = {}
upfile = {}
params = {}
_df_all_ev={}
_df_all = {}

class PrepData(object):
    def __init__(self, inputdir, outdir, VARS, skipexisting=False):
        self.path = inputdir
        self.outdir = outdir
        self.df_all = {}
        self.VARS   = VARS
        if skipexisting: 
            shutil.rmtree(self.outdir)
            os.makedirs(self.outdir)
        elif not os.path.exists(outdir):
            os.makedirs(self.outdir)
        #self.saveDF_ = saveDF
    
    # this is a function to look for specific pattern in directory
    # and then get back with the list of matched files 
    def find_all_matching(self,substring):
        self.result = []
        for root, dirs, files in os.walk(self.path):
            for thisfile in files:
                if substring in thisfile:
                    self.result.append(os.path.join(root, thisfile))

    def saveCSV(self):
        all_files = self.find_all_matching(".root")
        csv_dir = self.outdir
        sig_files = [ x for x in self.result if 'T1tttt' in x ]
        bkg_files = [ x for x in self.result if not 'T1tttt' in x ] 

        # check if the DFs are already in place  
        if os.path.exists(csv_dir+'/MultiClass_background.csv'):
            print ("background DF is already in place, no need to produce it again ")
            self.df_all['bkg'] = pd.read_csv(csv_dir+'/MultiClass_background.csv',index_col=None) 
        else:
          self.df_all['bkg'] = pd.DataFrame()

        if os.path.exists(csv_dir+'/MultiClass_signal.csv'): 
            print ("signal DF is already in place, no need to produce it again ")
            self.df_all['sig'] = pd.read_csv(csv_dir+'/MultiClass_signal.csv',index_col=None) 
        else:
            self.df_all['sig'] = pd.DataFrame()

        if self.df_all['bkg'].empty: 
            print ("self.df_all['bkg'] is empty i will look for the input root files to convert them")
            df = pd.DataFrame()
            for b in bkg_files: 
                #for block in it:
                #if "genMET" in b : continue 
                print(b) 
                it = uproot.open(b)["sf/t"]
                p_df = it.pandas.df(self.VARS+["Xsec"])
                p_df['filename'] = np.array(b.split("/")[-1].replace(".root","").replace("evVarFriend_","").replace("_ext",""))
                bkg_df = pd.concat([p_df, df], ignore_index=True)
                df = pd.concat([p_df, df], ignore_index=True)

            bkg_df = bkg_df.loc[~(bkg_df['filename'].isin(['TTJets_SingleLeptonFromT','TTJets_DiLepton','TTJets_SingleLeptonFromTbar'])) |
                            ((bkg_df['filename'].isin(['TTJets_SingleLeptonFromT','TTJets_DiLepton','TTJets_SingleLeptonFromTbar'])) &
                            (bkg_df['GenMET'] < 150 ))|
                            ((bkg_df['filename'].isin(['TTJets_SingleLeptonFromT_genMET','TTJets_DiLepton_genMET','TTJets_SingleLeptonFromTbar_genMET'])) &
                            (bkg_df['GenMET'] > 150 ))]
                                                        
            self.df_all['bkg'] =  bkg_df.loc[(bkg_df['nLep'] == 1) & (bkg_df['Lep_pt'] > 25)& (bkg_df['Selected'] == 1)& (bkg_df['Lep_pt'] > 25)&
                                    (bkg_df['nVeto'] == 0)& (bkg_df['nJets30Clean'] >= 5)& (bkg_df['Jet2_pt'] > 80)&
                                    (bkg_df['HT'] > 500)& (bkg_df['LT'] > 250)&(bkg_df['nBJet'] >= 1)]
            # cleanup not needed DFs
            del df
            del p_df
            del bkg_df

            # place the final weight (Xsec * all other SFs / total sum of weights)
            self.df_all['bkg'].loc[:,'Finalweight'] \
              = self.df_all['bkg'].Xsec \
              * self.df_all['bkg'].btagSF \
              * self.df_all['bkg'].puRatio \
              * self.df_all['bkg'].lepSF \
              * self.df_all['bkg'].nISRttweight \
              * self.df_all['bkg'].genWeight \
              / self.df_all['bkg'].sumOfWeights

            # drop unnecessary variables
            self.df_all['bkg'] = self.df_all['bkg'].drop(
                [
                  'sumOfWeights',
                  'genWeight',
                  'nISRttweight',
                  'Xsec',
                  'btagSF',
                  'lepSF',
                  'puRatio'
                ],
                axis=1
            )

        if self.df_all['sig'].empty: 
            print (
                "self.df_all['sig'] is empty",
                "i will look for the input root files to convert them"
            )
            dfs =  pd.DataFrame()
            for s in sig_files: 
                if "TuneCP2" in s: continue 
                if "evVarFriend_SMS_T1ttttCP5_MVA" in s: continue 
                if '22_points' in s: continue 

                #if '_15_01' in s : continue
                #for block in it:
                print(s) 
                its = uproot.open(s)["sf/t"]
                #print it.arrays(VARS)
                p_dfs = its.pandas.df(self.VARS + [
                    'susyXsec',
                    'mGo',
                    'mLSP',
                    'susyNgen',
                    'nISRweight'
                ])
                p_dfs['filename'] = \
                  np.array(s.split("/")[-1] \
                    .replace(".root","") \
                    .replace("evVarFriend_","") \
                    .replace("_ext",""))

                sig_df = pd.concat([p_dfs, dfs], ignore_index=True)
                dfs = pd.concat([p_dfs, dfs], ignore_index=True)

            self.df_all['sig'] =  sig_df.loc[(sig_df['nLep'] == 1) & (sig_df['Lep_pt'] > 25)& (sig_df['Selected'] == 1)& (sig_df['Lep_pt'] > 25)&
                                    (sig_df['nVeto'] == 0)& (sig_df['nJets30Clean'] >= 5)& (sig_df['Jet2_pt'] > 80)&
                                    (sig_df['HT'] > 500)& (sig_df['LT'] > 250)&(sig_df['nBJet'] >= 1)]

            # rename the column susXsec to Xsec 
            self.df_all['sig'].rename(columns={'susyXsec':'Xsec'},inplace=True)
            # cleanup not needed DFs
            del dfs
            del p_dfs 
            del sig_df
            # place the final weight (Xsec * all other SFs / total sum of weights)
            self.df_all['sig'].loc[:,'Finalweight'] \
              = self.df_all['sig'].Xsec \
              * self.df_all['sig'].btagSF \
              * self.df_all['sig'].puRatio \
              * self.df_all['sig'].lepSF \
              * self.df_all['sig'].nISRttweight \
              * self.df_all['sig'].genWeight \
              / self.df_all['sig'].susyNgen
              # / self.df_all['sig'].sumOfWeights

            # drop unnecessary variables
            self.df_all['sig'] \
              = self.df_all['sig'].drop(
                  [
                    'sumOfWeights',
                    'genWeight',
                    'nISRttweight',
                    'nISRweight',
                    'susyNgen',
                    'Xsec',
                    'btagSF',
                    'lepSF',
                    'puRatio'
                  ],
                  axis=1
            )

        # rearrange the the bkg to match with sig df 
        self.df_all['bkg'] = \
          self.df_all['bkg'].reindex(columns=self.df_all['sig'].columns)

        # Save the signals/bkg from .csv
        if not os.path.exists(csv_dir+'/MultiClass_background.csv') : 
            self.df_all['bkg'].to_csv(csv_dir+'/MultiClass_background.csv',index=None)
        if not os.path.exists(csv_dir+'/MultiClass_signal.csv') :
            self.df_all['sig'].to_csv(csv_dir+'/MultiClass_signal.csv',index=None)

