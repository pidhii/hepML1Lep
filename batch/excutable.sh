#!/bin/bash

eval 'export PATH="/nfs/dust/cms/user/amohamed/anaconda3/bin:$PATH"'

eval 'export KERAS_BACKEND=tensorflow'

source /nfs/dust/cms/user/amohamed/anaconda3/bin/activate hepML

cd $4

/nfs/dust/cms/user/amohamed/anaconda3/envs/hepML/bin/python evaluate.py --infile $1 --outdir $2 --model $3 --mGo $5 --mLSP $6
