# hepML1Lep
The package is to train and evaluate (binary or multiclass) neural network and/or xgb BDT,... within the context of susy single lepton analysis
The workflow will be 
 - prepare the training set of root trees into pandas dataframe (dfs) and split them into training/testing df
 - train and neural network with a specific/multiple signal model with/without doing parametric training i.e. give the physical identity of the model as a parameter in the training step, for this we use oversampling method to assign the parameter to the background
 - during the training and testing, we do some performance plots 
 - we use independent datasets for evaluation and farther analysis steps, to avoid any potential bias 

this package will take root files and append a DNN score and the very end to the root files

the setup is based on Anaconda 2019.03 for Linux Installer (https://www.anaconda.com/distribution/) for python3 

on DESY NAF one can install it by using 
 - ```bash /nfs/dust/cms/user/amohamed/Anaconda3-2019.03-Linux-x86_64.sh```
 - ```export PATH="path/to/anaconda3/bin:$PATH```
 - I keep everything as default but the installation dir i change it to a place where I have enough space
 - ```conda create -n hepML -c nlesc root=6.16 root_numpy  pandas seaborn scikit-learn matplotlib root_pandas uproot python=3.6.8```
 - ```conda activate hepML```
 - if you got any error related to "libstdc" and "libcrypto" when opening root I do : 
     - ```ln -s  path/to/anaconda3/envs/hepML/lib/libstdc++.so.6.0.26 path/to/anaconda3/envs/hepML/lib/libstdc++.so```
     - ```ln -s  path/to/anaconda3/envs/hepML/lib/libstdc++.so.6.0.26 path/to/anaconda3/envs/hepML/lib/libstdc++.so.6```
     - same for 'libcrypto'
 - if you have GPU and you want to use, you need to install tensorflow, tensorflow-gpu and keras but from `pip` as conda tensorflow is not doing the correct setup for `GPU`
 - ```pip install tensorflow tensorflow-gpu keras```

test the env by opening `python` and check if the python version is `3.6.8` and you can do : 
 - ```import ROOT```
 - ```import keras```
 - ```import tensorflow as tf```
 - ```sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))```

- an example to run the training and testing is `run.py` to prepare the dataframes, do training and testing, performance plots and save the model
- `evaluate.py` is prepared to evaluate the model on any of the independent samples we use for farther analysis
- `evaluate_onbatch.py` will wrap `evaluate.py` to run an independent batch system job for each sample, it will produce either `.root` or `.csv` based on the input file extension and it can also save the score only or save the entire sample based on what you need. Finally, it can run with parametric evaluation i.e. evaluate an indepenedet score for each signal hypothis  
