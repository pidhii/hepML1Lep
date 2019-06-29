#!/usr/bin/env python

from MLClass.eval import eval
import argparse

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a NAF batch system for nanoAOD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--infile', help='inputFile', metavar='infile')
    parser.add_argument('--outdir', help='output directory',default=None, metavar='outdir')
    parser.add_argument('--model', help='name of the model with out extensions',default=None, metavar='model')
    parser.add_argument('--mGo', help=' gluino mass ',default=0.0, metavar='mGo')
    parser.add_argument('--mLSP', help='LSP mass',default=0.0, metavar='mLSP')
    

    args = parser.parse_args()
    filename = args.infile
    outdir = args.outdir
    model = args.model
    mGo = args.mGo
    mLSP = args.mLSP

    ev = eval(filename,outdir,model,mGo = mGo , mLSP = mLSP)

    if '.root' in filename : 
        ev.ev_score_toROOT(savepredicOnly=True)
    elif '.csv' in filename : 
        ev.ev_score_toDF(savepredicOnly=True)
    
