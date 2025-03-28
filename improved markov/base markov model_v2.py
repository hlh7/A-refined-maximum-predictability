#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:04:55 2024

@author: luy
"""

import numpy as np
import pandas as pd

def time1(x):
    mm=x.split('_')
    ww=int(mm[2])
    hh=int(mm[3])
    if ww>4:
        nhh=hh+24
    else:
        nhh=hh
    return nhh

def time2(x):
    mm=x.split('_')
    ww=int(mm[2])
    if ww>4:
        nhh=1
    else:
        nhh=0
    return nhh

def markov_pred(juzhen1,d1):
    od=np.sum(juzhen1,axis=2)
    if np.sum(od[d1,:])>0:
        pred1=np.argsort(od[d1,:])[::-1][0:10].tolist()
    else:
        pred1=([-1])*10
    return pred1
        
func1=lambda x:time1(x)
func2=lambda x:time2(x)
func8=lambda x:[int(i) for i in x.split('_')]

flag='foursquare_nyc'

name1='../data_preprocessing/base_'+flag+'_v1.txt'

ff=open(name1,encoding='utf-8')
   
outname1='./'+flag+'_base_markov.txt'
g1=open(outname1,'w',encoding='utf-8')
acc=[]
onum=0
for ij in ff.readlines():
    mess=ij.strip().split('#')
    binds=mess[-1].split('&')
    
    user=int(mess[0])
    print(user)
    if 'test' in binds:
        dseq1=mess[1].split('&')
        tseq1=mess[2].split('&')
        ###########
        tseq2=list(map(func1,tseq1))
        tseq3=list(map(func2,tseq1))
        #######
        pdnuml=sorted(set(dseq1))
        pdnuml=pdnuml+(['-1','-2','-3','-4','-5','-6','-7','-8','-9','-10'])
        pdnum=len(pdnuml)
        
        tdnuml2=np.unique(tseq2)
        tdnum2=len(tdnuml2)
        
        tdnuml3=np.unique(tseq3)
        tdnum3=len(tdnuml3)
        
        juzhen1=np.zeros((pdnum,pdnum,tdnum2))
        juzhen2=np.zeros((pdnum,pdnum,tdnum3))
        
        pdd={pdnuml[i]:i for i in range(len(pdnuml))}
        tdd2={tdnuml2[i]:i for i in range(len(tdnuml2))}
        tdd3={tdnuml3[i]:i for i in range(len(tdnuml3))}
        
        tseq22=[tdd2[x] for x in tseq2]
        tseq33=[tdd3[x] for x in tseq3]
        
        dseq11=[pdd[x] for x in dseq1]
        
        top1=0
        top5=0
        top10=0
        numa=0
        ##########################################
        for j in range(1,len(dseq11)):
            t2=tseq22[j-1]#@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$
            t3=tseq33[j-1]
            
            d1=dseq11[j-1]
            d2=dseq11[j]
            
            if binds[j]=='test':
                numa=numa+1
                pred=markov_pred(juzhen1,d1)
                if d2 in pred:
                    ind=pred.index(d2)
                    if ind==0:
                        top1=top1+1
                        top5=top5+1
                    elif ind>0 and ind<5:
                        top5=top5+1
                    top10=top10+1
                    
            juzhen1[d1,d2,t2]=juzhen1[d1,d2,t2]+1
            juzhen2[d1,d2,t3]=juzhen2[d1,d2,t3]+1
        acc.append((top1,top5,top10,numa))
        

acc=np.array(acc)
t1=np.sum(acc[:,0])/np.sum(acc[:,3])
t2=np.sum(acc[:,1])/np.sum(acc[:,3])       
t3=np.sum(acc[:,2])/np.sum(acc[:,3])
print(t1,t2,t3,np.sum(acc[:,3]))
        
        
            
