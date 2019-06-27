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

def masslist(masslist):
    mass_list = []
    mGo = -999
    mLSP = -999 
    MASS_file = open(masslist,'r')
    for line in MASS_file : 
        small_list = []
        if line.startswith("#") : continue 
        line_ = line.strip()
        #print (line.split(" "))
        mGo = line_.split(" ")[0]
        mLSP = line_.split(" ")[-1]
        if (float(mGo) > 1900 and float(mGo) < 2300 ):
            if (float(mLSP)) > 1000 : continue
        elif (float(mGo) < 1900 and float(mGo) > 1400) : 
            if (float(mLSP) < 1000 or float(mLSP) > 1350) : continue 
        else : continue 
        small_list.append(mGo)
        small_list.append(mLSP)
        #print (small_list)
        mass_list.append(small_list)
    return mass_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a NAF batch system for nanoAOD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', help='List of datasets to process', metavar='indir')
    parser.add_argument('--outdir', help='output directory',default=None, metavar='outdir')
    parser.add_argument('--exec', help="wight directory", metavar='exec')
    parser.add_argument('--model', help='name of the model with out extensions',default=None, metavar='model')
    parser.add_argument('--mult', help=' set it to true if you want to evaluate paramtric training for each mass point',default=False, action='store_true')

    
    args = parser.parse_args()
    dirname = args.indir
    outdir = args.outdir
    execu = args.exec
    logdir = outdir+'/Logs' 
    model = args.model
    mult = args.mult

    wdir = os.getcwd()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir) 
    
    schedd = htcondor.Schedd()  

    Filenamelist = find_all_matching(".root",dirname)

    if not mult : 
        for fc in Filenamelist : 
            ##Condor configuration
            submit_parameters = { 
                "executable"                : execu,
                "arguments"                 : " ".join([fc, outdir,model,wdir,0,0]),
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
    elif mult : 
        mlist = masslist('./mass_list.txt')
        for mass in mlist:
            mGo = mass[0]
            mLSP = mass[1]
            for fc in Filenamelist : 
                ##Condor configuration
                submit_parameters = { 
                    "executable"                : execu,
                    "arguments"                 : " ".join([fc, outdir,model,wdir,mGo,mLSP]),
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


