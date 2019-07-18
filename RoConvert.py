#!/usr/bin/env python
#Convert a signal and background dataframe to a root tree
import os
import ROOT as r
from array import array
import pandas as pd
from root_numpy import root2array, tree2array
import argparse
from root_pandas import to_root
#Function to convert a file path to a tree to a dataframe
def convertTree(tree,signal=False,passFilePath=False,tlVectors=[]):
    if passFilePath:
        if isinstance(tree,list):
            chain = r.TChain('outtree')
            for t in tree:
                chain.Add(t)
            tree=chain
        else:
            rfile = r.TFile(tree) 
            tree = rfile.Get('outtree')
    #Note that this step can be replaced by root_pandas
    # this can also flatten the trees automatically
    df = pd.DataFrame(tree2array(tree))
    if len(tlVectors)>0: addTLorentzVectors(df,tree,branches=tlVectors)
    return df

def convert1LepDFs(infile,outdir,scores = []):
    # choose this to load as it has a feature of applying selection while loading the file
    list_ = ['TTS','TTDi','WJ','sig']
    df = pd.read_csv(infile,index_col=None)
    for i , score in enumerate(scores) : 
        if 'Logs' in score : continue 
        #print (score)
        #print (os.path.join(score,infile.split("/")[-1].replace('.csv','_'+score.split('/')[-1]+'.csv')))
        scoreDF = infile.split("/")[-1].replace('.csv','_'+score.split('/')[-1]+'.csv')
        names = [score.split('/')[-1]+X for X in list_]
        print (os.path.join(score,scoreDF))
        S_df = pd.read_csv(os.path.join(score,scoreDF),index_col=None,names=names,skiprows=1)
        df = pd.concat([df,S_df],axis=1, sort=False)
    df.to_root(outdir+'/'+infile.split('/')[-1].replace('.csv','.root'), key='sf/t')    
    del df
def find_all_matching(substring, path):
    result = []
    for root, dirs, files in os.walk(path):
        for thisfile in files:
            if substring in thisfile:
                result.append(os.path.join(root, thisfile ))
    return result

#Run on its own for testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a NAF batch system for nanoAOD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', help='List of datasets to process',default=None, metavar='indir')
    parser.add_argument('--infile', help='infile to process',default=None, metavar='infile')
    parser.add_argument('--scores', help='path to score dir where you have one dir for each mass score with name mGo_mLSP',default=None, metavar='scores')
    parser.add_argument('--outdir', help='output directory', metavar='outdir')
    parser.add_argument('--exec', help="wight directory", default='./batch/Roconv_exec.sh', metavar='exec')
    parser.add_argument('--batchMode','-b', help='Batch mode.',default=False, action='store_true')

    #parser.add_argument('--ana','-A', help='which analysis you want delphes or 1Lep skimmed tree, [1Lep,Delp]',default='1Lep',  metavar='ana')

    
    args = parser.parse_args()
    dirname = args.indir
    outdir = args.outdir
    execu = args.exec
    logdir = outdir+'/Logs' 
    batch = args.batchMode
    infile = args.infile
    scores = os.listdir(args.scores)
    scores = [os.path.join(args.scores,x) for x in scores if not x.startswith('.')]
    #ana = args.ana
    wdir = os.getcwd()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir) 
    
    if ((batch) and (dirname is not None)): 
        import htcondor
        schedd = htcondor.Schedd()  

        Filenamelist = find_all_matching(".csv",dirname) 
        print (Filenamelist)
        for fc in Filenamelist : 
            ##Condor configuration
            submit_parameters = { 
                "executable"                : execu,
                "arguments"                 : " ".join([fc,outdir,wdir,args.scores]),
                "universe"                  : "vanilla",
                "should_transfer_files"     : "YES",
                "log"                       : "{}/job_$(Cluster)_$(Process).log".format(logdir),
                "output"                    : "{}/job_$(Cluster)_$(Process).out".format(logdir),
                "error"                     : "{}/job_$(Cluster)_$(Process).err".format(logdir),
                "when_to_transfer_output"   : "ON_EXIT",
                'Requirements'              : 'OpSysAndVer == "CentOS7"',

             }
            job = htcondor.Submit(submit_parameters)
            with schedd.transaction() as txn:
                    job.queue(txn)
                    print ("Submit job for file {}".format(fc))
    if not batch : 
        convert1LepDFs(infile ,outdir,scores)
