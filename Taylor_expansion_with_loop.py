#import ROOT as rt
#import functions as f 
#import CMS_Lumi
#from ROOT import *
import uproot
import pandas
import numpy as np
import numpy.random as rand
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras import layers, models, losses, regularizers,optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from keras import backend as K
from keras.models import model_from_json
from sklearn.decomposition import PCA
import heapq

#tf.enable_eager_execution()

#load the keras model from json

json_file = open ('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights into new model
loaded_model.load_weights("model_weights.h5")
print("loaded model from disk")


#open root files with uproot

ZTT = uproot.open("/nfs/dust/cms/user/mameyer/SM_HiggsTauTau/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_DYJets.root")

ggh = uproot.open ("/nfs/dust/cms/user/mameyer/SM_HiggsTauTau/HTauTau_emu/Inputs/NTuples_2018/em-NOMINAL_ntuple_ggH.root")

# read branches from Root as Numpy array

tree_ZTT = ZTT["TauCheck"]

tree_ZTT.keys()
Background=tree_ZTT.pandas.df(["m_sv","jpt_1","jpt_2","njets","mjj","jdeta","dijetpt","pt_vis","pt_1","pt_2","m_vis","ME_q2v1","ME_q2v2","mTdileptonMET", "xsec_lumi_weight","iso_1","iso_2","metFilters","trg_muonelectron","extraelec_veto","extramuon_veto","q_1","q_2","nbtag" ,"isZTT"])
#print('ZTT')
#print(Background)

selection = Background.query('pt_1 > 15 and pt_2> 15 and (pt_1 >24|pt_2 >24) and iso_1<0.15 and iso_2<0.2  and metFilters and trg_muonelectron and extraelec_veto<0.5 and extramuon_veto<0.5 and q_1*q_2 < 0 and nbtag==0  and mTdileptonMET<60 and isZTT==True')
#print('Back_after_selection')
#print(selection)

Back_selec = selection.loc[:,'m_sv':'xsec_lumi_weight']
#print ('Back after selection')
#print(Back_selec)

tree_ggh = ggh["TauCheck"]
tree_ggh.keys()
signal=tree_ggh.pandas.df(["m_sv","jpt_1","jpt_2","njets","mjj","jdeta","dijetpt","pt_vis","pt_1","pt_2","m_vis","ME_q2v1","ME_q2v2","mTdileptonMET" , "xsec_lumi_weight", "iso_1", "iso_2", "metFilters","trg_muonelectron","extraelec_veto","extramuon_veto","q_1","q_2","nbtag","isZTT"])

#print('ggh',signal)

selection_sig = signal.query('pt_1 > 15 and pt_2> 15 and (pt_1 >24|pt_2 >24) and iso_1<0.15 and iso_2<0.2  and  metFilters and trg_muonelectron and extraelec_veto<0.5 and extramuon_veto<0.5 and q_1*q_2 < 0 and nbtag==0 and mTdileptonMET<60')

sig_selec = selection_sig.loc[:,'m_sv':'xsec_lumi_weight']
#print ('signal after selection')
#print (sig_selec)

#labeling background to 0 signal to 1

label_Background = np.zeros(len(Back_selec))
Back = np.column_stack ((Back_selec,label_Background))
#print('Backround_label',Back)

label_signal = np.ones(len(sig_selec))
sig = np.column_stack ((sig_selec, label_signal))
#print('Signal_label', sig)


Data = np.concatenate((sig,Back),axis=0)  
#print('Data',Data)

Data_ran = shuffle(Data,random_state =1)
print('Data_random',Data_ran)

x= Data_ran[:,0:14]
rest= Data_ran[:,14:15]

#preprocessing of the input data

scaler= StandardScaler()
standardized = scaler.fit_transform(x)
inverse = scaler.inverse_transform(standardized)

#split arrays into random train and test subsets with train test split 

X= standardized

wei = np.column_stack ((X,rest))
print('wei',wei)
print(wei.shape)


Y= Data_ran[:,-1]

seed=1

(X_train_L, X_test_L, Y_train, Y_test) = train_test_split(wei ,Y , train_size= 0.75 ,test_size=0.25, random_state=seed)
X_train=X_train_L[:,:-1]
X_test = X_test_L [:,0:14]
X_test_lumi = X_test_L [:,-1]



(X_train, X_val, Y_train, Y_val) = train_test_split(X_train,Y_train,test_size=0.25, random_state=seed)
opt = optimizers.Adam(lr=0.0001)
pca = PCA(n_components = 14)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_val = pca.transform(X_val)


loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
results =loaded_model.evaluate(X_test,Y_test,batch_size= 256)
print('test_loss,test acc:',results)

#define input in tensorflow

x=tf.placeholder(dtype=tf.float32, shape= [None,14])
print('x',x)
input_x = X_test.astype(np.float32) 
print('input_x', input_x)
print('m_sv',input_x[0:1])
print('jpt_1',input_x[:,2])
print('jpt_2',input_x[:,3])

#write the NN in tensorflow

w1= loaded_model.layers[0].get_weights()[0]
b1 = loaded_model.layers[0].get_weights()[1]
w2= loaded_model.layers[1].get_weights()[0]
b2 = loaded_model.layers[1].get_weights()[1]
w3= loaded_model.layers[2].get_weights()[0]
b3 = loaded_model.layers[2].get_weights()[1]
#w4= loaded_model.layers[3].get_weights()[0]
#b4 = loaded_model.layers[3].get_weights()[1]
#print('w1', w1,'w2',w2)
#print('b1',b1,'b2',b2)


W1= tf.get_variable("w1",initializer = w1, dtype=np.float32)
B1 = tf.get_variable("b1",initializer=b1, dtype=np.float32)
W2= tf.get_variable("w2",initializer= w2, dtype=np.float32)
B2= tf.get_variable("b2",initializer=b2,dtype=np.float32)
W3= tf.get_variable("w3",initializer= w3, dtype=np.float32)
B3= tf.get_variable("b3",initializer=b3,dtype=np.float32)
#W4= tf.get_variable("w4",initializer= w4, dtype=np.float32)
#B4= tf.get_variable("b4",initializer=b4,dtype=np.float32)



#calculate the outputof the layers
hidden_out = tf.tanh(tf.add(tf.matmul(x,W1),B1))
hidden_out1 = tf.tanh(tf.add(tf.matmul(hidden_out,W2),B2))
#hidden_out2 = tf.tanh(tf.add(tf.matmul(hidden_out1,W3),B3))
                                            
#calculate the output of NN 

y=tf.sigmoid(tf.add(tf.matmul(hidden_out1, W3),B3))
print('y',y)
print(type(y))
print(np.size(y))

#define the first and second order gradient

first_order_gradients = []
second_order_gradients = []


with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output=sess.run(y,feed_dict={x:input_x})
            #print('output',output)
            
            for i in range(14):
                (index,first_order) = (i, (tf.gradients(y,x)[0])[:,i:i+1])
                A = sess.run(first_order, feed_dict = {y:output, x:input_x})
                
                #index = {'0':'m_sv', '1':'jpt_1','2':'jpt_2','3':'njets','4':'mjj','5':'jdeta','6':'dijetpt','7':'pt_vis','8':'pt_1','9':'pt_2','10':'m_vis','11':'ME_q2v1','12':'ME_q2v2','13':'mTdileptonmet' }
                #for value in index.values():
                print('First order Gradient wrt ' + str(index) ,np.average(np.absolute(A),axis= 0))
                first_order_gradients.append(np.average(np.absolute(A))) 
                #print('first_order_gradients',first_order_gradients)
                
                for j in range(i,14):
                    (index1, index2, second_order) = (i,j,(tf.gradients(first_order,x)[0])[:,j:j+1])
                 #calculate the gradient
                    B = sess.run(second_order, feed_dict = {x: input_x})
                #for vlaue in index.items():
                    #index_1 = index.get(str(i))
                    #index_2 = index.get(str(j))
                    print('Second order Gradient wrt '+ str(index1)+':'+str(index2),np.average(np.absolute(B),axis= 0))
                    second_order_gradients.append(np.average(np.absolute(B)))
                    #print('second_order_gradients',second_order_gradients)
                               
                               
            plt.figure(1)
            y = first_order_gradients
            x = range(len(y))
            plt.scatter(x,y, marker="o", s=11**2, color = 'green')
            plt.grid()
            plt.xticks(np.arange(0,14,step=1.0),('t_p0','t_p1','t_p2','t_p3','t_p4','t_p5','t_p6','t_p7','t_p8','t_p9','t_p10','t_p11','t_p12','t_p13'))
            plt.xticks(fontsize= 10,rotation= 40)
            plt.ylim(0.0,0.40)
            plt.ylabel('<ti>')
            plt.title('Taylor')
            name=('T_First_orders_loop_200epoch_pca')
            plt.savefig('Taylor_loop/{}'.format(name))
            plt.show()
           
            plt.figure(2)
            y1 = second_order_gradients
            maxvalues = heapq.nlargest(20, y1)
            print('20_top',maxvalues)
            x = range(len(maxvalues))
            plt.figure(figsize=(14,6.0))
            plt.scatter(x,maxvalues, marker="o", s=11**2, color = 'green')
            plt.grid()
            plt.xticks(np.arange(0,21,step=1.0),('tp2_p2','tp2_p5','tp5_p5','tp6_p6','tp12_p12','tp2_p6','tp2_p12','tp10_p10','tp2_p13','tp1_p2','tp5_p6','tp6_p12','tp1_p6','tp0_p6','tp5_p12','tp1_p5','tp1_p1','tp0_pt0','tp9_p10','tp5_p13'))
           
            plt.xticks(fontsize= 9,rotation= 40)
            plt.ylim(0.0,0.60)
            plt.ylabel('<ti>')
            plt.title('Taylor')
            name=('T_Second_orders_loop_200epoch_pca')
            plt.savefig('Taylor_loop/{}'.format(name))
            plt.show()                   
                               
                               
                               
                               