#!/usr/bin/env python
#Convert a signal and background root tree to a dataframe
import os
import ROOT as r
from array import array
import pandas as pd
from root_numpy import root2array, tree2array
import argparse

#Add the important components of the tlorentzvectors to the dataframe
def addTLorentzVectors(df,tree,branches):
    nentries=tree.GetEntries()
    arrDict = {}
    #Initialise empty arrays
    for b in branches:
        arrDict[b+'_pt'] = []
        arrDict[b+'_px'] = []
        arrDict[b+'_py'] = []
        arrDict[b+'_pz'] = []
        arrDict[b+'_phi'] = []
        arrDict[b+'_eta'] = []
        arrDict[b+'_m'] = []
        arrDict[b+'_e'] = []

    #Loop over the tree and fill the arrays with the info from the vectors
    for i,event in enumerate(tree):
        for b in branches:
            arrDict[b+'_pt'].append( [o.Pt() for o in getattr(event,b)])
            arrDict[b+'_px'].append( [o.Px() for o in getattr(event,b)])
            arrDict[b+'_py'].append( [o.Py() for o in getattr(event,b)])
            arrDict[b+'_pz'].append( [o.Pz() for o in getattr(event,b)])
            arrDict[b+'_phi'].append( [o.Phi() for o in getattr(event,b)])
            arrDict[b+'_eta'].append( [o.Eta() for o in getattr(event,b)])
            arrDict[b+'_m'].append( [o.M() for o in getattr(event,b)])
            arrDict[b+'_e'].append( [o.E() for o in getattr(event,b)])
        pass
    pass

    #Add the info to the data frames
    for b in branches:
        df[b+'_pt'] = arrDict[b+'_pt'] 
        df[b+'_px'] = arrDict[b+'_px'] 
        df[b+'_py'] = arrDict[b+'_py'] 
        df[b+'_pz'] = arrDict[b+'_pz'] 
        df[b+'_phi'] = arrDict[b+'_phi']
        df[b+'_eta'] = arrDict[b+'_eta']
        df[b+'_m'] = arrDict[b+'_m']
        df[b+'_e'] = arrDict[b+'_e']
             
    pass

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

def convert1LepTree(infile,outdir,flatten=False,select="nLep >= 1 && nJets30Clean >= 1 && HT > 500 && LT > 250"):
    # choose this to load as it has a feature of applying selection while loading the file
    # convert the root into numpy array
    x = root2array(infile,treename='sf/t',selection=select)
    y =pd.DataFrame.from_records(x.tolist(), columns=x.dtype.names)
    # trick to flatten tree when it has mnay variables with many indeces
    if flatten :
        for name, column in y.items():
            if column.dtype == object:
                column = y[name].apply(pd.Series)
                y = y.drop(name, axis=1)
                column = column.rename(columns = lambda x : name +'_'+ str(x))
                y = pd.concat([y[:], column[:]], axis=1)
    y.to_csv(outdir+'/'+infile.split('/')[-1].replace('.root','.csv'),index=None)    
    
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
    parser.add_argument('--outdir', help='output directory', metavar='outdir')
    parser.add_argument('--exec', help="wight directory", default='./batch/dfconv_exec.sh', metavar='exec')
    parser.add_argument('--batchMode','-b', help='Batch mode.',default=False, action='store_true')

    
    args = parser.parse_args()
    dirname = args.indir
    outdir = args.outdir
    execu = args.exec
    logdir = outdir+'/Logs' 
    batch = args.batchMode
    infile = args.infile

    wdir = os.getcwd()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir) 
    
    
    if ((batch) and (dirname is not None)): 
        import htcondor
        schedd = htcondor.Schedd()  

        Filenamelist = find_all_matching(".root",dirname) 
        print (Filenamelist)
        for fc in Filenamelist : 
            ##Condor configuration
            submit_parameters = { 
                "executable"                : execu,
                "arguments"                 : " ".join([fc,outdir,wdir]),
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
        convert1LepTree(infile,outdir,flatten=True,select="nLep >= 1 && nJets30Clean >= 1 && HT > 500 && LT > 250")
