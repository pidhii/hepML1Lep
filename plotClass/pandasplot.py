import ROOT as rt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

fig_style = {
    'figsize': (5, 5),
}

ratio_fig_style = {
    'figsize': (5, 5),
    'gridspec_kw': {'height_ratios': (3, 1)},
}

stack_style = {
    'edgecolor': (0, 0, 0, 0.5),
}

hatch_style = {
    'facecolor': 'none',
    'edgecolor': (0, 0, 0, 0.5),
    'linewidth': 0,
    'hatch': '///',
}

errorbar_style = {
    'linestyle': 'none',
    'marker': '.',
    'elinewidth': 1,
    'capsize': 1,
    'color': 'k',
}

shaded_style = {
    'facecolor': (0,0,0,0.3),
    'linewidth': 0
}

class pandasplot(object) : 
    def __init__(self,outdir,var,cuts=None,weights = None) : 
        self.outdir     = outdir 
        self.var        = var    
        self.cuts       = cuts   
        self.weights    = weights
        self.outputplot=os.path.join(self.outdir,'plots')
        if not os.path.exists(self.outputplot): os.makedirs(self.outputplot)

    def makecuts(sefl,DFlist=[])  : 
        pass
    def make1D(sefl):
        pass
    def make2D(self):
        pass
    def makeCanv(self) : 
        pass
    def makeratio(self) : 
        pass
    def styling(self):
        pass
    def error1D(self,ax) : 
        pass
    def var_plot(self,testclf,trainclf,norm=True,logY=False,append='',multiclass=True,class_names=[]):
        
        for feature in self.var:
            if multiclass:
                print ("multiclass classifier plots will be made")
                fig, ax = plt.subplots()
                for i in range(0,len(class_names)):
                    print(i)
                    fig, ax = plt.subplots()
                    class_test_w = np.full(shape=testclf.query('isSignal=='+str(i))[feature].shape,fill_value=len(trainclf.query('isSignal=='+str(i))[feature].index)/len(testclf.query('isSignal=='+str(i))[feature].index),dtype=np.float32)
                    Non_class_test_w = np.full(shape=testclf.query('isSignal!='+str(i))[feature].shape,fill_value=len(trainclf.query('isSignal!='+str(i))[feature].index)/len(testclf.query('isSignal!='+str(i))[feature].index),dtype=np.float32)
                    print ("weight train / test = ", class_test_w)
                    _ = ax.hist(trainclf.query('isSignal=='+str(i))[feature], bins=30, alpha=0.5, density=norm, color='b'  , weights=trainclf.query('isSignal=='+str(i))['Finalweight'] ,log=logY,label='class '+str(i)+' (train)')
                    _ = ax.hist(testclf.query('isSignal=='+str(i))[feature] , bins=30, alpha=0.5, density=norm, color='r'  , weights=class_test_w*testclf.query('isSignal=='+str(i))['Finalweight'] ,log=logY,label='class '+str(i)+' (test)')
                    _ = ax.hist(trainclf.query('isSignal!='+str(i))[feature], bins=30, alpha=0.5, density=norm, color='b'  , weights=trainclf.query('isSignal!='+str(i))['Finalweight'] , histtype='step' ,log=logY,label='non class '+str(i)+' (train)')
                    _ = ax.hist(testclf.query('isSignal!='+str(i))[feature] , bins=30, alpha=0.5, density=norm, color='r'  , weights=Non_class_test_w*testclf.query('isSignal!='+str(i))['Finalweight'], histtype='step' ,log=logY,label='non class '+str(i)+' (test)')
                    ax.set_title("DNN Class "+str(i))
                    ax.legend(loc='best')
                    plt.savefig(os.path.join(self.outputplot,'MultiClass_'+feature+append+'_'+str(i)+".png"))
                    plt.clf()
                    plt.close('all')
            else : 
                print ("binary classifier plots will be made")
                fig, ax = plt.subplots()
                bkg_w = np.full(shape=testclf.query('isSignal==0')[feature].shape,fill_value=len(trainclf.query('isSignal==0')[feature].index)/len(testclf.query('isSignal==0')[feature].index),dtype=np.float32)
                sig_w = np.full(shape=testclf.query('isSignal==1')[feature].shape,fill_value=len(trainclf.query('isSignal==1')[feature].index)/len(testclf.query('isSignal==1')[feature].index),dtype=np.float32)
                _ = ax.hist(trainclf.query('isSignal==0')[feature] , bins=30, alpha=0.5, density=norm, color='b'  , weights=trainclf.query('isSignal==0')['Finalweight']   ,log=logY, label='background (train)')
                _ = ax.hist(testclf.query('isSignal==0')[feature]  , bins=30, alpha=0.5, density=norm, color='r'  , weights=bkg_w*testclf.query('isSignal==0')['Finalweight']   ,log=logY, label='background (test)')
                _ = ax.hist(trainclf.query('isSignal==1')[feature] , bins=30, alpha=0.5, density=norm, color='b'  , weights=trainclf.query('isSignal==1')['Finalweight']   ,histtype='step' ,log=logY, label='Signal (train)')
                _ = ax.hist(testclf.query('isSignal==1')[feature]  , bins=30, alpha=0.5, density=norm, color='r'  , weights=sig_w*testclf.query('isSignal==1')['Finalweight']   ,histtype='step' ,log=logY, label='Signal (test)')
                ax.set_title(feature)
                ax.legend(loc='best')
                plt.savefig(os.path.join(self.outputplot,feature+append+".png"))
                plt.clf()
                plt.close('all')
                
    def classifierPlot(self,testclf,trainclf,norm=True,logY=False,append='',multiclass=True):
        # select slice of the DF (only classifier out , labels)
        testclf = testclf.loc[:, 'isSignal':]
        trainclf = trainclf.loc[:, 'isSignal':]
        #print(testclf)
        print (len(testclf.columns))
        if multiclass:
            print ("multiclass classifier plots will be made")
            fig, ax = plt.subplots()
            for i in range(0,len(testclf.columns)):
                print(i)
                # skip first column as it has the labels only 
                if i == 0 : continue 
                fig, ax = plt.subplots()
                class_test_w = np.full(shape=testclf.query('isSignal=='+str(i-1))[i-1].shape,fill_value=len(trainclf.query('isSignal=='+str(i-1))[i-1].index)/len(testclf.query('isSignal=='+str(i-1))[i-1].index),dtype=np.float32)
                Non_class_test_w = np.full(shape=testclf.query('isSignal!='+str(i-1))[i-1].shape,fill_value=len(trainclf.query('isSignal!='+str(i-1))[i-1].index)/len(testclf.query('isSignal!='+str(i-1))[i-1].index),dtype=np.float32)
                print ("weight train / test = ", class_test_w)
                _ = ax.hist(trainclf.query('isSignal=='+str(i-1))[i-1], bins=30, alpha=0.5, density=norm, color='b'  , weights=None                 ,log=logY,label='class '+str(i-1)+' (train)')
                _ = ax.hist(testclf.query('isSignal=='+str(i-1))[i-1] , bins=30, alpha=0.5, density=norm, color='r'  , weights=class_test_w         ,log=logY,label='class '+str(i-1)+' (test)')
                _ = ax.hist(trainclf.query('isSignal!='+str(i-1))[i-1], bins=30, alpha=0.5, density=norm, color='b'  , weights=None                 , histtype='step' ,log=logY,label='non class '+str(i-1)+' (train)')
                _ = ax.hist(testclf.query('isSignal!='+str(i-1))[i-1] , bins=30, alpha=0.5, density=norm, color='r'  , weights=Non_class_test_w     , histtype='step' ,log=logY,label='non class '+str(i-1)+' (test)')
                ax.set_title("DNN Class "+str(i-1))
                ax.legend(loc='best')
                plt.savefig(os.path.join(self.outputplot,"DNN_"+str(i-1)+append+".png"))
                plt.clf()
                plt.close('all')
        else : 
            print ("binary classifier plots will be made")
            fig, ax = plt.subplots()
            bkg_w = np.full(shape=testclf.query('isSignal==0')[0].shape,fill_value=len(trainclf.query('isSignal==0')[0].index)/len(testclf.query('isSignal==0')[0].index),dtype=np.float32)
            sig_w = np.full(shape=testclf.query('isSignal==1')[0].shape,fill_value=len(trainclf.query('isSignal==1')[0].index)/len(testclf.query('isSignal==1')[0].index),dtype=np.float32)
            _ = ax.hist(trainclf.query('isSignal==0')[0] , bins=30, alpha=0.5, density=norm, color='b'  , weights=None   ,log=logY, label='background (train)')
            _ = ax.hist(testclf.query('isSignal==0')[0]  , bins=30, alpha=0.5, density=norm, color='r'  , weights=bkg_w   ,log=logY, label='background (test)')
            _ = ax.hist(trainclf.query('isSignal==1')[0] , bins=30, alpha=0.5, density=norm, color='b'  , weights=None   ,histtype='step' ,log=logY, label='Signal (train)')
            _ = ax.hist(testclf.query('isSignal==1')[0]  , bins=30, alpha=0.5, density=norm, color='r'  , weights=sig_w   ,histtype='step' ,log=logY, label='Signal (test)')
            ax.set_title("DNN Class")
            ax.legend(loc='best')
            plt.savefig(os.path.join(self.outputplot,"DNN_binary"+append+".png"))
            plt.clf()
            plt.close('all')