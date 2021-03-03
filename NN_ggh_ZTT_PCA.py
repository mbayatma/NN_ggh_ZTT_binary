
import ROOT as rt
#import functions as f 
#import CMS_Lumi
from ROOT import *
import uproot
import pandas as pd
import numpy as np
import numpy.random as rand
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras import layers, models, losses, regularizers, optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std

#open root files with uproot

ZTT = uproot.open("/nfs/dust/cms/user/mameyer/SM_HiggsTauTau/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_DYJets.root")

ggh = uproot.open ("/nfs/dust/cms/user/mameyer/SM_HiggsTauTau/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_ggH.root")

# read branches from Root as Numpy array

tree_ZTT = ZTT["TauCheck"]

tree_ZTT.keys()
#Background=tree_ZTT.pandas.df(["m_sv","jpt_1","jpt_2","njets","mjj","jdeta","dijetpt","pt_vis","pt_1","pt_2","m_vis","ME_q2v1","ME_q2v2","mTdileptonMET","xsec_lumi_weight","iso_1","iso_2","metFilters","trg_muonelectron","extraelec_veto","extramuon_veto","q_1","q_2","nbtag","isZTT",])

Background=tree_ZTT.pandas.df(["m_sv","jpt_1","pt_vis","jdeta","m_vis","d0_1","d0_2","dr_tt","met","eta_1","pt_sv","pt_tt","pt_ttjj","xsec_lumi_weight","iso_1","iso_2","metFilters","trg_muonelectron","extraelec_veto","extramuon_veto","q_1","q_2","nbtag","isZTT",])



print('ZTT')
print(Background)

selection = Background.query('iso_1<0.15 and iso_2<0.2  and metFilters and trg_muonelectron and extraelec_veto<0.5 and extramuon_veto<0.5 and q_1*q_2 < 0 and nbtag==0  and isZTT==True')
print('Back_after_selection')
print(selection)

#define the weights
Back_selec = selection.loc[:,'m_sv':'xsec_lumi_weight']
print ('Back after selection')
print(Back_selec)



#weight = tree_ZTT.pandas.df(["xsec_lumi_weight"])
#print('w',weight)

#WEI = (Back_selec)*(weight)
#print('WEI', WEI)

tree_ggh = ggh["TauCheck"]
tree_ggh.keys()
#signal=tree_ggh.pandas.df(["m_sv","jpt_1","jpt_2","njets","mjj","jdeta","dijetpt","pt_vis","pt_1","pt_2","m_vis","ME_q2v1","ME_q2v2","mTdileptonMET" ,"xsec_lumi_weight", "iso_1", "iso_2", "metFilters","trg_muonelectron","extraelec_veto","extramuon_veto","q_1","q_2","nbtag", "isZTT"])

signal=tree_ggh.pandas.df(["m_sv","jpt_1","pt_vis","jdeta","m_vis","d0_1","d0_2","dr_tt","met","eta_1","pt_sv","pt_tt","pt_ttjj" ,"xsec_lumi_weight", "iso_1", "iso_2", "metFilters","trg_muonelectron","extraelec_veto","extramuon_veto","q_1","q_2","nbtag", "isZTT"])

#print('ggh',signal)

selection_sig = signal.query(' iso_1<0.15 and iso_2<0.2  and  metFilters and trg_muonelectron and extraelec_veto<0.5 and extramuon_veto<0.5 and q_1*q_2 < 0 and nbtag==0  ')

sig_selec = selection_sig.loc[:,'m_sv':'xsec_lumi_weight']
print ('signal after selection')
print (sig_selec)


#labeling background to 0 signal to 1

label_Background = np.zeros(len(Back_selec))
Back = np.column_stack ((Back_selec,label_Background))
print('Backround_label',Back)

label_signal = np.ones(len(sig_selec))
sig = np.column_stack ((sig_selec, label_signal))
print('Signal_label', sig)


Data = np.concatenate((sig,Back),axis=0)  
print('Data',Data)

Data_ran = shuffle(Data,random_state =1)
print('Data_random',Data_ran)
print('shape of data_ran', Data_ran.shape)
print('dimension', Data_ran.ndim)



#split arrays into random train and test subsets with train test split 
x= Data_ran[:,0:13]
#return the Lumi weight Column
rest= Data_ran[:,13:14]
#print('rest',rest)
#print('dATA-RAN',x)
Y= Data_ran[:,-1]
#print('Y of Data_ran',Y)
#preprocessing of the input data

scaler= StandardScaler()
standardized = scaler.fit_transform(x)
#inverse = scaler.inverse_transform(standardized)

#split arrays into random train and test subsets with train test split 

X= standardized
#print('X',X)
#print(X.shape)

wei = np.column_stack ((X,rest))
print('wei',wei)
print(wei.shape)

Y= Data_ran[:,-1]

# How to calculate PCAs mathematicaly and before i found out PCA is defined in sklearn so just the PCA.fit works!
# calculating cov matrix eigen_vectors and values of X

cov_mat = np.cov(X.T)

print('Covariance Matrix \n%s' %cov_mat)
print('shape of cov ' ,cov_mat.shape)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_vals_cor = np.corrcoef (eig_vals)
print('corr eig vals', eig_vals_cor)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print('eig_pairs',eig_pairs)
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#choosing the PCA for new feature space

tot = sum(eig_vals)

# explained variance
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print('var_exp',var_exp)
cum_var_exp = np.cumsum(var_exp)
print('cum_var_exp', cum_var_exp)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(1)

    plt.bar(range(13), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(13), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    name= ('PCA')
    plt.savefig('PCA/{}'.format(name))


#matrix_w = np.hstack((eig_pairs[0][1].reshape(14,1),
                     #eig_pairs[1][1].reshape(14,1)))

#print('Matrix W:\n', matrix_w)
#print('X',X)
#print(X.shape)

#wei = np.column_stack ((X,rest))
#print('wei',wei)




seed=1

(X_train_L, X_test_L, Y_train, Y_test) = train_test_split(wei ,Y , train_size= 0.70 ,test_size=0.30, random_state=seed)
#print(X_train,Y_train)
X_train=X_train_L[:,:-1]
X_test = X_test_L [:,0:13]
X_test_lumi = X_test_L [:,-1]

(X_train, X_val, Y_train, Y_val) = train_test_split(X_train,Y_train,test_size=0.25, random_state=seed)
print('X_train_before_PCA',X_train[0,])

pca = PCA(n_components = 13)
#print('pca components',pca)

X_train = pca.fit_transform(X_train)
#print('X_train_after_PCA', X_train[0])

#inverse of PCA to observe the inputs!
#nComp = 14
#Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
#mu = np.mean(X,axis= 0)
#Xhat += mu
#print('Xhat',Xhat[0])

X_test = pca.transform(X_test)
#print('X_test_after_pca', X_test)



#x_test_cor = np.corrcoef (X_test)
#print('corr after pca', x_test_cor)

X_val = pca.transform(X_val)


model = models.Sequential (
        [ 
            layers.Dense(100,activation= 'tanh', input_dim = 13 ,kernel_regularizer = regularizers.l2(0.0001)),
            #layers.Dropout(rate=0.3),
            #layers.Dense(100, activation= 'tanh', kernel_regularizer = regularizers.l2(0.0001)),
            #layers.BatchNormalization(),
            #layers.Dropout(rate= 0.3),
            #layers.Dense(20, activation= 'tanh', kernel_regularizer = regularizers.l2(0.0001)),
            #layers.BatchNormalization(),
            #layers.Dense(200, activation= 'tanh', kernel_regularizer = regularizers.l2(0.0001)),
            #layers.Dropout(rate= 0.3),
            layers.Dense(1,activation = 'sigmoid')
     
     
            ]
        )
model.summary()
#print('X_test',X_test,'Y_test',Y_test)



class_weights = class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train)

#class_weights = {0:1.44,1:0.76} 

earlystopping_callback= EarlyStopping(monitor='val_acc',patience=10,verbose=1, min_delta=0.0001)
#opt = optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy',optimizer='Adam', metrics=['accuracy'])
print(model.optimizer.get_config())

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200, batch_size=256,class_weight= class_weights, verbose = 2)
#callbacks=[earlystopping_callback],



#store the model architecture
#json_dict = model.to_json()
#with open("model.json","w") as json_file:
     #json_file.write(json_dict)
    
#serialize weights to hdf5
#model.save_weights("model_weights.h5")
#print("saved model to disk")

#plot the learning rate
#plt.figure(5)
#nb_epoch = len(history.history['loss'])
#learning_rate=history.history['lr']
#xc=range(nb_epoch)
#plt.figure(3,figsize=(7,5))
#plt.plot(xc,learning_rate)
#plt.xlabel('num of Epochs')
#plt.ylabel('learning rate')
#plt.title('Learning rate')
#plt.grid(True)
#plt.style.use(['seaborn-ticks'])
#plt.savefig('NN_ZTT_GGH_PLOTS_2layers_100node_200 epoch/{}'.format(name))

# ACC plot versus PCA components

# get the dataset
#def get_dataset():
    #X, Y = make_classification(n_samples=1000, n_features=13, n_informative=11, n_redundant=2, random_state=7)
    #return X, Y
 
    #get a list of models to evaluate
#def get_models():
    #models = dict()
    #for i in range(1,13):
        #steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
        #models[str(i)] = Pipeline(steps=steps)
        #return models
    #evaluate a given model using cross-validation
#def evaluate_model(model):
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #scores = cross_val_score(model, X_val, Y_val, scoring='accuracy', cv=cv, n_jobs=-1 )#, error_score='raise')
    #return scores
 
# define dataset
#X, y = get_dataset()
# get the models to evaluate
#models = get_models()
# evaluate the models and store results
#results, names, list_mean_scores = list(), list(), list()
#for name, model in models.items():
    #scores = evaluate_model(model)
    #print('scores',scores)
    #results.append(scores)
    #print('results',results)
   
    #names.append(name)
    #print('names',names)
    #list_mean_scores = []
    #list_mean_scores.append(mean(scores))
    #print('list_mean_scores', list_mean_scores)
    #print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

    # plot model performance for comparison

#plt.figure(6)
#plt.boxplot(results, labels=names, showmeans=False)
#plt.scatter(names, list_mean_scores ,color= 'red',marker="o",s=11**2 )
#plt.xticks(rotation=0)
#plt.ylabel('accuracy')
#plt.xlabel('number of PCA components')
#plt.grid()
#name=('acc_PCA_not_boxplot_14pca_newvar')
#plt.savefig('PCA_new/{}'.format(name))

#acc plot
plt.figure(2)
plt.plot(history.history['acc'])#, color = 'orange')
plt.plot(history.history['val_acc'])#, color = 'blue')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'],loc= 'upper left')
name=('acc_plot_100nodes_2layers_70_30_neuvar')
plt.savefig('PCA_new/{}'.format(name))
plt.show()
   
#loss plot
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc= 'upper left')
name_2= ('loss_plot_100nodes_2layers_70_30_neuvar')
plt.savefig('PCA_new/{}'.format(name_2))
plt.show()


#plot Roc curve
plt.figure(4)
#X_test =  pca.transform(X_test)
probs = model.predict_proba(X_test)
preds = probs[:,0]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
name= ('Roc_Curve_100nodes_2layers_70_30_neuvar')
plt.savefig('PCA_new/{}'.format(name))
plt.show()

#Probability plot
plt.figure(5)
Z=model.predict(X_test)
plt.figure(figsize=(12,7))
plt.hist(Z[Y_test==0],bins=40,label='Background',weights=X_test_lumi[Y_test==0])
lumisig=X_test_lumi[Y_test==1]
plt.hist(Z[Y_test==1],bins=40,label='Signal X 100',alpha=0.7,color='r',weights=lumisig*100)
plt.xlabel('Probability of being Signal or Background')
plt.ylabel('Number of records in each bucket',fontsize=20)
plt.legend(fontsize=15)
name= ('Prob_Plot_100nodes_2layers_70_30_neuvar')
plt.savefig('PCA_new/{}'.format(name))
plt.show() 

#Signal over sqrt(B)
plt.figure(5)
plt.subplot(111)
(hist_bkg, bins_bkg,_) = plt.hist(Z[Y_test==0],bins=51,label='Background',weights=X_test_lumi[Y_test==0])
lumisig=X_test_lumi[Y_test==1]
(hist_sig, bins_sig,_) = plt.hist(Z[Y_test==1],bins=51,label='Signal X 100',alpha=0.7,color='r',weights=lumisig*100)
print('hist_sig', hist_sig)
print('hist_bkg', hist_bkg)

Stot = hist_sig.sum() / 100

Btot = hist_bkg.sum()

hist_ratio = hist_sig
S=0
B=0
for bin in range(51):
    S+=hist_sig[bin]/100
    print('s',S)
    B+=hist_bkg[bin]
    print('B',B)
    print('bin',bin)
    #if B < 0.0001: 
        #hist_ratio[bin] = 0
    #else:
    hist_ratio[bin] = (Stot-S)/np.sqrt(Btot-B+Stot-S)
    
    
print('hist_ratio',hist_ratio)
print('len_hist_ratio', len(hist_ratio))
plt.figure(6)
plt.xlabel('NN score cut value')
plt.ylabel('$\\frac{S}{\\sqrt{S+B}}$')
hist_ratio = hist_ratio[~np.isnan(hist_ratio)]
#plt.bar(np.arange(0,1,1/40),hist_ratio, width= (1/40))
plt.plot (np.arange(0,1,0.02), hist_ratio )

x = np.arange(0,1,0.02)
y = hist_ratio

fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_max(x,y)


ax.set_ylim(0.9,2.5)
plt.xlabel('NN score cut value')
plt.ylabel('$\\frac{S}{\\sqrt{S+B}}$')
plt.show()
name=('signal significance 100nodes 2layers 70 30 neuvar')
plt.savefig('PCA_new/{}'.format(name))

