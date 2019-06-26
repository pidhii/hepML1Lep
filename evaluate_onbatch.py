#!/usr/bin/env python
import sys,os
#sys.path.append("/nfs/dust/cms/user/amohamed/anaconda3/envs/hepML/lib/python3.6/site-packages/")
import htcondor
import argparse

def find_all_matching(substring, path):
    result = []
    for root, dirs, files in os.walk(path):
        for thisfile in files:
            if substring in thisfile:
                result.append(os.path.join(root, thisfile ))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a NAF batch system for nanoAOD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', help='List of datasets to process', metavar='indir')
    parser.add_argument('--outdir', help='output directory',default=None, metavar='outdir')
    parser.add_argument('--exec', help="wight directory", metavar='exec')
    parser.add_argument('--model', help='name of the model with out extensions',default=None, metavar='model')
    
    args = parser.parse_args()
    dirname = args.indir
    outdir = args.outdir
    execu = args.exec
    logdir = outdir+'/Logs' 
    model = args.model

    wdir = os.getcwd()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir) 
    
    schedd = htcondor.Schedd()  
    Filenamelist = find_all_matching(".root",dirname)
    for fc in Filenamelist : 
        ##Condor configuration
        submit_parameters = { 
            "executable"                : execu,
            "arguments"                 : " ".join([fc, outdir,model,wdir]),
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
