#!/usr/bin/env python

import getopt
import sys
import ROOT
import pandas as pd

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
    print("     --loss         <loss-function> = 'None'")
    print("     --extra-layers <N-extra-layers> = 0")
    print("     --batch-size   <batch-size> = 1024")
    print("     --multiclass")
    print("     --load-model   <model-path>")
    print("     --learn-rate   <learn-rate> = 0.0001")
    print("     --test")
    print("     --lsp-mass     <mass of LSP>")
    print("     --batch-mode")
    print("     --class-weights")
    quit(err)

# Mandatory arguments.
CSV_DIR = None
DATA_DIR = "/nfs/dust/cms/user/amohamed/susy-desy/CMGSamples/FR_forMVA_nosplit_resTop/"
WORK_DIR = None
OUT_DIR = None
EPOCHS = 100
LOSS = 'None'
EXTRA_LAYERS = 0
BATCH_SIZE = 1024
MULTICLASS = False
MODEL_PATH = None
LEARN_RATE = 0.0001
TEST = False
MLSP = 1000
BATCHMODE = False
MONITOR = 'val_loss'
CLASSWEIGHTS = False

lumi = 30. #luminosity in /fb
# SIG = 17.6*0.059*lumi #cross section of stop sample in fb times efficiency measured by Marco
#expectedSignal = 228.195*0.14*lumi #leonid's number
# BG = 844000.*8.2e-4*lumi #cross section of ttbar sample in fb times efficiency measured by Marco
# SYS = 0.1 #systematic for the asimov signficance
LUMI = 35.9E+03

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
    "learn-rate=",
    "test",
    "lsp-mass=",
    "batch-mode",
    "monitor=",
    "class-weights"
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
    elif opt in ('--test'):
        TEST = True
    elif opt in ("--lsp-mass"):
        MLSP = int(arg)
    elif opt in ("--batch-mode"):
        BATCHMODE = True
    elif opt in ("--monitor"):
        MONITOR = arg
    elif opt in ('--class-weights'):
        CLASSWEIGHTS = True
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
print("{}: {}".format(orange("Mlsp"), MLSP))
print("{}: {}".format(orange("monitor"), MONITOR))
if MODEL_PATH is not None:
    print("{}: {}".format(blue("model"), MODEL_PATH))
if TEST:
    print(blue("test mode"))
if CLASSWEIGHTS:
  print(blue("use class weights"))
if not input("confirm? ").lower() in ("y", "yes"):
    print("aborting");
    exit(0)

################################################################################
# Main
#
print("\x1b[38;5;3;1m---\x1b[0m import ML-modules")
import preperData
from preperData.splitDFs import splitDFs
from preperData.PrepData import PrepData
from MLClass.score import score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
# copied from A.Elwood https://github.com/aelwood/hepML/blob/master/MlFunctions/DnnFunctions.py
from MlFunctions.DnnFunctions import significanceLoss,significanceLossInvert,significanceLoss2Invert,significanceLossInvertSqrt,significanceFull,asimovSignificanceLoss,asimovSignificanceLossInvert,asimovSignificanceFull,truePositive,falsePositive
import MlFunctions.DnnFunctions as loss

preperData.splitDFs.set_Mlsp(MLSP)

# Validate loss-function.
print("\x1b[38;5;3;1m---\x1b[0m validate loss")
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
print("\x1b[38;5;3;1m---\x1b[0m prepare data")
Data = PrepData(DATA_DIR, CSV_DIR, VARS, skipexisting = False)
Data.saveCSV()
# print("\x1b[38;5;3;1m---\x1b[0m data:")
# print(list(Data.df_all['sig'].columns.values))
# quit(0)

# preper the data and split them into testing sample + training sample
print("\x1b[38;5;3;1m---\x1b[0m split data")
splitted = splitDFs(
    Data.df_all['sig'],
    Data.df_all['bkg'],
    do_multiClass = MULTICLASS,
    nSignal_Cla = 1,
    do_parametric = True,
    split_Sign_training = False
)
splitted.prepare()
splitted.split(splitted.df_all['all_sig'], splitted.df_all['all_bkg'])

##########################
# init the modele 
print("\x1b[38;5;3;1m---\x1b[0m initialize model")
scoreing = score(
    WORK_DIR,
    splitted.test_DF,
    splitted.train_DF,
    splitted.class_weights if CLASSWEIGHTS else None,
    var_list = var_list,
    do_multiClass = MULTICLASS,
    nSignal_Cla = 1,
    do_parametric = True,
    split_Sign_training = False,
    class_names = class_names,
    monitor = MONITOR
)

# Build the model or load the pretrained one.
if loadmodel: 
    print("\x1b[38;5;3;1m---\x1b[0m load model at \"{}\"".format(MODEL_PATH))
    scoreing.load_model(MODEL_PATH, loss=LOSS)
else: 
    print("\x1b[38;5;3;1m---\x1b[0m build model")
    scoreing.build(multi=MULTICLASS, nclass=len(class_names), loss=LOSS, dropout=True, extra_layers=EXTRA_LAYERS)

if TEST:
    # Evaluate DNN in test-mode.
    # print("\x1b[38;5;3;1m---\x1b[0m test model")
    # scoreing.eval(batch_size=BATCH_SIZE)
    pass
else:
    # Train.
    print("\x1b[38;5;3;1m---\x1b[0m training")
    scoreing.train(epochs=EPOCHS, batch_size=BATCH_SIZE)
    # Save it.
    print("\x1b[38;5;3;1m---\x1b[0m save the model")
    scoreing.save_model(scoreing.model)

##########################
# start the performance plottng 
# 1- the DNN score plots
print("\x1b[38;5;3;1m---\x1b[0m get model predictions")
train_s_df = pd.DataFrame(scoreing.score_train())
test_s_df = pd.DataFrame(scoreing.score_test())

print("train_s_df:", train_s_df)
print("test_s_df:", test_s_df)

print("\x1b[38;5;3;1m---\x1b[0m import plotting modules")
from plotClass.pandasplot import pandasplot
import plots
# from ROOT import TCanvas, TH1F, TGraph, TLegend
# from math import log, sqrt
# import numpy as np

###############################################################################
# Classifier Output plot
#
print("\x1b[38;5;3;1m---\x1b[0m plotting classifier output")
classplot = plots.Classifier(test_s_df, scoreing, multi = MULTICLASS)

a = classplot.signal().GetBinContent(0)
b = classplot.signal().Integral()
c = classplot.signal().GetBinContent(plots.NBINS+1)
print("total signal: {} + {} + {} = {}".format(a, b, c, a + b + c))

a = classplot.background().GetBinContent(0)
b = classplot.background().Integral()
c = classplot.background().GetBinContent(plots.NBINS+1)
print("total background: {} + {} + {} = {}".format(a, b, c, a + b + c))

classplot.prepare()
classplot.draw()


###############################################################################
# Asimov significance vs Cut on Classifier output.
#
print("\x1b[38;5;3;1m---\x1b[0m plotting asimov significance vs cut on classifier output")
asiplot = plots.Significance(classplot.signal(), classplot.background())
asiplot.prepare()
asiplot.draw()


full_test = pd.concat([scoreing.testDF,test_s_df],axis=1)
full_train = pd.concat([scoreing.trainDF,train_s_df],axis=1)
plott = pandasplot(OUT_DIR, var_list)
plott.classifierPlot(full_test,full_train,norm=False,logY=True,append='',multiclass=MULTICLASS)

if not TEST:
    plott.var_plot(full_test,full_train,norm=False,logY=True,append='',multiclass=MULTICLASS,class_names=class_names)

if not TEST:
    # 2- the DNN loss and acc plotters 
    scoreing.performance_plot(scoreing.history,scoreing.score_test(),scoreing.score_train(),append=append)

# 3- the DNN ROC plotters 
if MULTICLASS: 
    scoreing.rocCurve_multi(scoreing.score_test(),label_binarize(splitted.test_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Test'+append,n_classes=4)
    scoreing.rocCurve_multi(scoreing.score_train(),label_binarize(splitted.train_DF['isSignal'], classes=[0,1,2,3]),append='MultiClass_Train'+append,n_classes=4)
else: 
    scoreing.rocCurve(scoreing.score_test(),label_binarize(splitted.test_DF['isSignal'], classes=[0,1]),append='Binary_Test')
    scoreing.rocCurve(scoreing.score_train(),label_binarize(splitted.train_DF['isSignal'], classes=[0,1]),append='Binary_Train')

# 4- the DNN confusion matrix plotters 
test_cm = confusion_matrix(splitted.test_DF["isSignal"],scoreing.score_test().argmax(axis=1))
train_cm = confusion_matrix(splitted.train_DF["isSignal"],scoreing.score_train().argmax(axis=1))

scoreing.plot_confusion_matrix(test_cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',append="test"+append)
scoreing.plot_confusion_matrix(train_cm, classes=class_names, normalize=True,
                               title='Normalized confusion matrix', append="train"+append)

# 5- the DNN correlation matrix plotters 
scoreing.heatMap(splitted.test_DF, append=append)
##########################

if not BATCHMODE: ROOT.gApplication.Run()
