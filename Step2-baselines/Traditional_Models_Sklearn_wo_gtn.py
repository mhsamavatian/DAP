import pandas as pd
import numpy as np
import random
import os
import sys
import psutil

import math
from multiprocessing import cpu_count,Pool 
import multiprocessing

from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier as GBC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='Atlanta')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
CITY     = args.city
REPEATS  = args.repeats

cores = cpu_count() #Number of CPU cores on your system
partitions = cores
SEQ=8

class WithExtraArgs(object):
    def __init__(self, func, **args):
        self.func = func
        self.args = args
    def __call__(self, df):
        return self.func(df, **self.args)

def applyParallel(data, func,pool,partition, kwargs):
    data_split = [data[i:i + partition] for i in xrange(0, len(data), partition)]
    #data_split = np.array_split(data, min(partitions,data.shape[0]))
    data =pool.map(WithExtraArgs(func, **kwargs), data_split)
    #data = pd.concat(pool.map(WithExtraArgs(func, **kwargs), data_split))
    return data

def parallelize(data, func,pool,partition):
    data_split = [data[i:i + partition] for i in xrange(0, len(data), partition)]
    #data_split = np.array_split(data, partitions)
    data =pool.map(func, data_split)
    return data

def reshape_cat(array,category):
    l=[]
    b = array[:,0:-14]
    if category!='geohash' and  category!='NLP' :
        for i in range(SEQ):
            c = b[:,i*25:i*25+25]
            if category == 'traffic':
                d = np.concatenate([c[:,1:2],c[:,3:10]],axis=1)
            elif category=='weather':
                d = c[:,10:-5]
            elif category=='time':
                d = np.concatenate([c[:,0:1],c[:,2:3],c[:,-5:]],axis=1) 
            else:
                d = c
            l.append(d)        
        n = np.concatenate(l,axis=1)
        return n
    elif category=='NLP':
        return array[:,-100:]
    else:
        return array[:,-114:-100]

class base_model(object): 
       
    def __init__(self,n_jobs=-1,metric='f1_score'): 
        self.n_jobs=n_jobs
        if metric == 'precision':
            self.metric = make_scorer(precision_score, average= 'weighted')
        elif metric =='recall':
            self.metric = make_scorer(recall_score, average= 'weighted')
        elif metric =='f1_score':
            self.metric = make_scorer(f1_score, average= 'weighted')
        else:
            print ('not valid metric')
        pass 
   
    def load_data(self,category=None):
        print ('load and test: shapes for train and test X and Y')
        self.X_train = np.load('train_set/X_train_'+CITY+'.npy')[:,0:-1]
            
        self.y_train = np.load('train_set/y_train_'+CITY+'.npy')
            
        self.X_test = np.load('train_set/X_test_'+CITY+'.npy')[:,0:-1]
            
        self.y_test = np.load('train_set/y_test_'+CITY+'.npy')
                    
            
        if category!=None:
            l_train=[]
            l_test=[]
            for cat in category:
                l_train.append(reshape_cat(self.X_train,cat))
                l_test.append(reshape_cat(self.X_test,cat))
            self.X_train = np.concatenate(l_train,axis=1)
            self.X_test = np.concatenate(l_test,axis=1)
                        
        print (self.X_train.shape)
        print (self.y_train.shape)
        print (self.X_test.shape)
        print (self.y_test.shape)
        
    def train(self):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        
        
        self.clf = GridSearchCV(self.model, self.tuned_parameters, cv=cv, scoring=self.metric,n_jobs=self.n_jobs)
        self.clf.fit(self.X_train, self.y_train)
        print("Best parameters set found on development set:")
        print ()
        print(self.clf.best_params_)
        print("Grid scores on development set:")
        print ()
        means = self.clf.cv_results_['mean_test_score']
        stds = self.clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, self.clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        return self.clf.best_params_
       
    def evaluate(self):
        
        y_true, y_pred = self.y_test, self.clf.predict(self.X_test)
        print(classification_report(y_true, y_pred))
        dict_out = classification_report(y_true, y_pred,output_dict=True)
        return dict_out

class model_LR_SKlearn(base_model):
    def create_model(self):
        self.tuned_parameters = [{'penalty': ['l2'],'max_iter':[100,1000,10000,100000],'solver':['newton-cg', 'lbfgs' ,'sag'], 'n_jobs':[10]},
                                 {'penalty': ['l1'],'max_iter':[100,1000,10000,100000],'solver':['liblinear', 'saga'],'n_jobs':[1]}]
        self.model = LogisticRegression(verbose=0)

class model_GBC_SKlearn(base_model):
    def create_model(self):
        self.tuned_parameters = {'learning_rate':[0.1,0.15,0.05,0.01],'n_estimators': [100,200,300,400], 
                                 'max_depth': [3,4,5,6]}
        self.model = GBC(n_iter_no_change=15)

def make_models(city='Atlanta',model='LR',category=None,metric='precision'):
    CITY = city
    if model=='LR':
        mypred = model_LR_SKlearn(metric=metric)
    elif model=='GBC':
        mypred = model_GBC_SKlearn(n_jobs=partitions,metric=metric)
    mypred.load_data(category)
    mypred.create_model()
    best_params = mypred.train()
    dict_out = mypred.evaluate()
    return pd.DataFrame(dict_out),best_params
    
models     = ['LR', 'GBC']
categories = ['time', 'weather']

for m in models:
    print '\n', m
    No_Acc   = []
    Acc      = []        
    Macro    = []
    Micro    = []
    Weighted = []
    
    for i in range(REPEATS):
        df,best_params = make_models(model=m, metric='f1_score', category=categories)        

        metrics_no_acc = list(df['0'])  # [f1, precision, recall, support (a.k.a sample size)]
        metrics_acc    = list(df['1'])
        macro_avg      = list(df['macro avg'])
        micro_avg      = list(df['micro avg'])
        weighted_avg   = list(df['weighted avg'])
        
        No_Acc.append(metrics_no_acc)
        Acc.append(metrics_acc)
        Macro.append(macro_avg)
        Micro.append(micro_avg)
        Weighted.append(weighted_avg)
        
    No_Acc    = np.mean(No_Acc, axis=0)
    Acc       = np.mean(Acc, axis=0)
    Macro     = np.mean(Macro, axis=0)
    Micro     = np.mean(Micro, axis=0)
    Weighted  = np.mean(Weighted, axis=0)
    
    writer = open('Results/Traditional_For_{}_{}_wo_gtn.csv'.format(CITY, m), 'w')
    writer.write('F1_Accident {}\n'.format(Acc[0]))
    writer.write('F1_No-Accident {}\n'.format(No_Acc[0]))
    writer.write('F1_WeightedAvg {}\n'.format(Weighted[0]))    
    writer.write('F1_Micro {}\n'.format(Micro[0]))    
    writer.write('F1_Macro {}\n'.format(Macro[0]))    
    writer.write('Precision_Accident {}\n'.format(Acc[1]))
    writer.write('Precision_No-Accident {}\n'.format(No_Acc[1]))
    writer.write('Precision_WeightedAvg {}\n'.format(Weighted[1]))
    writer.write('Recall_Accident {}\n'.format(Acc[2]))
    writer.write('Recall_No-Accident {}\n'.format(No_Acc[2]))
    writer.write('Recall_WeightedAvg {}\n'.format(Weighted[2]))
    writer.close()
