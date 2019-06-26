#!/usr/bin/env python

from MLClass.eval import eval
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a NAF batch system for nanoAOD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--infile', help='inputFile', metavar='infile')
    parser.add_argument('--outdir', help='output directory',default=None, metavar='outdir')
    parser.add_argument('--model', help='name of the model with out extensions',default=None, metavar='model')

    args = parser.parse_args()
    filename = args.infile
    outdir = args.outdir
    model = args.model

    ev = eval(filename,outdir,model)

    if '.root' in filename : 
        ev.ev_score_toROOT()
    elif '.csv' in filename : 
        ev.ev_score_toDF()
    
