#!/usr/bin/env python

import getopt
import sys

################################################################################
# Command Line Arguments
#
def helpAndExit(err = 0):
    print("usage: {} OPTIONS".format(sys.argv[0]))
    print("")
    print("\x1b[1;4mOPTIONS\x1b[0m:")
    print("")
    print("  \x1b[1mMandatory\x1b[0m")
    print("     --csv          <directory-for-csv-files>")
    print("  -D --dir          <working-directory>")
    print("")
    print("  \x1b[1mOptional\x1b[0m")
    print("     --data         <path-to-root-files>")
    print("  -o --outdir       <path-for-plots> = <working-directory>")
    print("     --epochs       <N-epochs> = 100")
    print("     --loss         <loss-function> = None")
    print("     --extra-layers <N-extra-layers> = 0")
    print("     --batch-size   <batch-size> = 4096")
    print("     --multiclass")
    print("     --load-model   <model-path>")
    print("     --learn-rate   <learn-rate> = 0.0001")
    quit(err)

# Mandatory arguments.
CSV_DIR = None
DATA_DIR = "/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/"
WORK_DIR = None
OUT_DIR = None
EPOCHS = 100
LOSS = None
EXTRA_LAYERS = 0
BATCH_SIZE = 1024
MULTICLASS = False
MODEL_PATH = None
LEARN_RATE = 0.0001

lumi = 30. #luminosity in /fb
SIG = 17.6*0.059*lumi #cross section of stop sample in fb times efficiency measured by Marco
#expectedSignal = 228.195*0.14*lumi #leonid's number
BG = 844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco
SYS = 0.1 #systematic for the asimov signficance

# Parse command line.
long_opts = [
    "help",
    "data=",
    "csv=",
    "dir=",
    "outdir=",
    "epochs=",
    "loss=",
    "extra-layers=",
    "batch-size=",
    "multiclass",
    "load-model=",
    "learn-rate="
];
try:
    opts, args = getopt.getopt(sys.argv[1:], "hD:o:", long_opts)
except getopt.GetoptError as exn:
    print("Error:", str(exn))
    print("")
    helpAndExit(1)

# Apply options.
for opt, arg in opts:
    if opt in ('-h', '--help'):
        helpAndExit(0)
    elif opt in ('--csv'):
        CSV_DIR = arg
    elif opt in ('--data'):
        DATA_DIR = arg
    elif opt in ('-D', '--dir'):
        WORK_DIR = arg
    elif opt in ('-o', '--outdir'):
        wOUT_DIR = arg
    elif opt in ('--epochs'):
        EPOCHS = int(arg)
    elif opt in ('--loss'):
        LOSS = arg
    elif opt in ('--extra-layers'):
        EXTRA_LAYERS = int(arg)
    elif opt in ('--batch-size'):
        BATCH_SIZE = int(arg)
    elif opt in ('--multiclass'):
        MULTICLASS = True
    elif opt in ('--load-model'):
        MODEL_PATH = arg
    elif opt in ('--learn-rate'):
        LEARN_RATE = float(arg)
    else:
        print("Error: undefined command line option,", opt)
        helpAndExit(1)

# Validate options.
if CSV_DIR is None:
    print("Error: path to directory for .csv-files not set")
    helpAndExit(1)

if WORK_DIR is None:
    print("Error: path to working-directory not set")
    helpAndExit(1)

# Output-directory defaults to working-directory.
OUT_DIR = OUT_DIR or WORK_DIR

# Dump configuration.
def orange(s): return "\x1b[38;5;202;1m" + s + "\x1b[0m"
def blue(s): return "\x1b[38;5;51;1m" + s + "\x1b[0m"
print("{}: {}".format(orange("data"), DATA_DIR))
print("{}: {}".format(orange("csv"), CSV_DIR))
print("{}: {}".format(orange("working directory"), WORK_DIR))
print("{}: {}".format(orange("output"), OUT_DIR))
print("{}: {}".format(orange("epochs"), EPOCHS))
print("{}: {}".format(orange("loss"), LOSS))
print("{}: {}".format(orange("N extra layers"), EXTRA_LAYERS))
print("{}: {}".format(orange("batch size"), BATCH_SIZE))
print("{}: {}".format(orange("multiclass"), MULTICLASS))
print("{}: {}".format(orange("learn rate"), LEARN_RATE))
if MODEL_PATH is not None:
    print("{}: {}".format(blue("model"), MODEL_PATH))
if not input("confirm? ").lower() in ("y", "yes"):
    print("aborting");
    exit(0)

################################################################################
# Main
#
from preperData.splitDFs import splitDFs
from preperData.PrepData import PrepData
from MLClass.score import score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
# copied from A.Elwood https://github.com/aelwood/hepML/blob/master/MlFunctions/DnnFunctions.py
from MlFunctions.DnnFunctions import significanceLoss,significanceLossInvert,significanceLoss2Invert,significanceLossInvertSqrt,significanceFull,asimovSignificanceLoss,asimovSignificanceLossInvert,asimovSignificanceFull,truePositive,falsePositive

# Validate loss-function.
try:
    LOSS = eval(LOSS);
except Exception as exn:
    print("Failed to resolve loss-function:", str(exn))
    exit(1)

# if you want to use a pretrained model activate it and give the model path wthout any extension
loadmodel = MODEL_PATH is not None
append=''

##########################
# multiclass or binary 
if MULTICLASS: 
    class_names = ['TTSemiLep','TTDiLep','WJets','signal']
else: 
    class_names = ['signal','background']

##########################
# variables to be used in the training 
var_list = ['MET', 'MT', 'Jet2_pt','Jet1_pt', 'nLep', 'Lep_pt', 'Selected',
            'nVeto', 'LT', 'HT', 'nBCleaned_TOTAL', 'nTop_Total_Combined',
            'nJets30Clean', 'dPhi', "Lep_relIso", "Lep_miniIso", "iso_pt",
            "iso_MT2", "mGo", "mLSP"]

# variables to be used in while transfere DFs
VARS = ["MET", "MT", "Jet2_pt", "Jet1_pt", "nLep", "Lep_pt", "Selected",
        "nVeto", "LT", "HT", "nBCleaned_TOTAL", "nBJet", "nTop_Total_Combined",
        "nJets30Clean", "dPhi", "met_caloPt", "lheHTIncoming",
        "genTau_grandmotherId", "genTau_motherId", "genLep_grandmotherId",
        "genLep_motherId", "DiLep_Flag", "semiLep_Flag", "genWeight",
        "sumOfWeights", "btagSF", "puRatio", "lepSF", "nISRttweight", "GenMET",
        "Lep_relIso", "Lep_miniIso", "iso_pt", "iso_MT2"]

##########################
# start preparing the data if it's not in place
Data = PrepData(DATA_DIR, CSV_DIR, VARS, skipexisting = False)
Data.saveCSV()

# preper the data and split them into testing sample + training sample
splitted = splitDFs(Data.df_all['sig'],Data.df_all['bkg'],do_multiClass = MULTICLASS,nSignal_Cla = 1,do_parametric = True,split_Sign_training = False)
splitted.prepare()
splitted.split(splitted.df_all['all_sig'],splitted.df_all['all_bkg'])

##########################
# init the modele 
scoreing = score(
    'DNN',
    WORK_DIR,
    splitted.test_DF,
    splitted.train_DF,
    splitted.class_weights,
    var_list=var_list,
    do_multiClass = MULTICLASS,
    nSignal_Cla = 1,
    do_parametric = True,
    split_Sign_training = False,
    class_names=class_names
)

# if continue pretrained model
if loadmodel: 
    append =' _2nd'
    # mode will be save automatically
    scoreing.load_model(MODEL_PATH, loss = LOSS, epochs = EPOCHS, learn_rate = LEARN_RATE)
else: 
    # nClass will be ignored in binary classification tasks anywayes
    # loss = None will use the normal cross entropy change it if you want to whatever defined in MlFunctions/DnnFunctions.py
    scoreing.do_train(
        nclass = len(class_names),
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        loss = LOSS,
        extra_layers = EXTRA_LAYERS
    )
    #scoreing.load_model()
    scoreing.save_model(scoreing.model) # here we need to enforce saving it

##########################
# start the performance plottng 
# 1- the DNN score plots
from plotClass.pandasplot import pandasplot
import pandas as pd
train_s_df = pd.DataFrame(scoreing.dnn_score_train)
test_s_df = pd.DataFrame(scoreing.dnn_score_test)
full_test = pd.concat([scoreing.testDF,test_s_df],axis=1)
full_train = pd.concat([scoreing.trainDF,train_s_df],axis=1)
plott = pandasplot(OUT_DIR, var_list)
plott.classifierPlot(full_test,full_train,norm=False,logY=True,append='',multiclass=MULTICLASS)
plott.var_plot(full_test,full_train,norm=False,logY=True,append='',multiclass=MULTICLASS,class_names=class_names)

# 2- the DNN loss and acc plotters 
scoreing.performance_plot(scoreing.history,scoreing.dnn_score_test,scoreing.dnn_score_train,append=append)

# 3- the DNN ROC plotters 
if MULTICLASS : 
    scoreing.rocCurve_multi(scoreing.dnn_score_test,label_binarize(splitted.test_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Test'+append,n_classes=4)
    scoreing.rocCurve_multi(scoreing.dnn_score_train,label_binarize(splitted.train_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Train'+append,n_classes=4)
else : 

    scoreing.rocCurve(scoreing.dnn_score_test,label_binarize(splitted.test_DF['isSignal'], classes=[0,1]),append='Binary_Test')
    scoreing.rocCurve(scoreing.dnn_score_train,label_binarize(splitted.train_DF['isSignal'], classes=[0,1]),append='Binary_Train')

# 4- the DNN confusion matrix plotters 
test_cm = confusion_matrix(splitted.test_DF["isSignal"],scoreing.dnn_score_test.argmax(axis=1))
train_cm = confusion_matrix(splitted.train_DF["isSignal"],scoreing.dnn_score_train.argmax(axis=1))

scoreing.plot_confusion_matrix(test_cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',append="test"+append)
scoreing.plot_confusion_matrix(train_cm, classes=class_names, normalize=True,
                               title='Normalized confusion matrix', append="train"+append)

# 5- the DNN correlation matrix plotters 
scoreing.heatMap(splitted.test_DF, append=append)
##########################
