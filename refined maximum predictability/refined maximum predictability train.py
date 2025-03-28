#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:50:56 2022

@author: luy
"""

import random
import numpy as np
import pandas as pd

def compute_f(p,S,N):
    if p<=0 or p>=1 :
        print(p)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    pi_max = h + (1 - p) * np.log2(N - 1) - S
    return pi_max

def getapproximation(p,S,N) :
    f= compute_f(p,S,N)
    d1 = np.log2(1-p) - np.log2(p) - np.log2(N-1)
    d2 = 1 / ((p-1)*p)
    return f/(d1-f*d2/(2*d1))

def unc_entropy(sequence):
    """
    Compute temporal-uncorrelated entropy (Shannon entropy).
    Equation:
    S_{unc} = - \sum p(i) \log_2{p(i)}, for each symbol i in the input sequence.
    Args:
        sequence: the input sequence of symbols.
    Returns:
        temporal-uncorrelated entropy of the input sequence.
    Reference:
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu,
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    _, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def max_predictability(S, N):
    """
    Estimate the maximum predictability of a sequence with
    entropy S and alphabet size N.
    Equation:
    $S = - H(\Pi) + (1 - \Pi)\log(N - 1),$
        where $H(\Pi)$ is given by
    $H(\Pi) = \Pi \log_2(\Pi) + (1 - \Pi) \log_2(1 - \Pi)$
    Args:
        S: the entropy of the input sequence of symbols.
        N: the size of the alphabet (number of unique symbols)
    Returns:
        the maximum predictability of the sequence.
    Reference:
        Limits of Predictability in Human Mobility. Chaoming Song, Zehui Qu,
        Nicholas Blumm1, Albert-László Barabási. Vol. 327, Issue 5968, pp. 1018-1021.
        DOI: 10.1126/science.1177170
    """
    if S>np.log2(N):
        return 0
    
    if S<=0.01:
        return 0.999
    
    p = (N+1)/(2*N)
    while(abs(compute_f(p,S,N))>0.0000001):
        p = p - 0.8*getapproximation(p,S,N)
    return p


def shannon(juzhen,mm):
    rzong=np.sum(juzhen)

    bizhi=juzhen/rzong
    
    juzhen2=np.log2(bizhi)*-1
    
    ff=juzhen2*bizhi
    ff[np.isnan(ff)]=0
    ff1=np.sum(ff)
    max_proba0 = max_predictability(ff1,mm)
    txt0=str(ff1)+'#'+str(max_proba0)
    return txt0

#### refined fusion condition entropy final version
def han_rce_od(juzhen,ax,ke,m1,m2,dd):
    channel1=np.sort(-1*juzhen,axis=ax)#m
    channel1=-1*channel1
    
    lmax=np.where(channel1[:,0]<=ke)
    
    channel11=np.zeros((m1,m2))
    channel11[lmax[0],:]=channel1[lmax[0],:]
    yizong=np.sum(channel11)
    channel1[lmax[0],:]=0
    
    zong1=np.sum(channel1)
    total=zong1+yizong
    if zong1>0:
        sum1=np.sum(channel1,axis=ax).reshape((m1,1))
        bb=sum1/total
        channel=channel1/sum1
        #######################
        channel7=np.log2(channel)*-1*channel
        channel7[np.isnan(channel7)]=0
        sum2=np.sum(channel7,axis=ax)
        ppset=[]
        for zu in range(len(sum2)):
            if sum1[zu,0]>0:
                dnum=np.where(channel1[zu,:]>0)
                pp=max_predictability(sum2[zu],len(dnum[0]))
                ppset.append(pp)
            else:
                ppset.append(0)
        ppset=np.array(ppset).reshape((-1,1))
        fpp1=np.sum(bb*ppset)
    else:
        fpp1=0
    fpp=round(fpp1+yizong/total/dd,5)
    return fpp
        
def han_rce_tod(juzhen,ax,ke,m,sdn,dd):
    channel1=np.sort(-1*juzhen,axis=ax)
    channel1=-1*channel1
    
    lmax=np.where(channel1[:,0,:]<=ke)
    
    channel11=np.zeros((m,m,sdn))
    channel11[lmax[0],:,lmax[1]]=channel1[lmax[0],:,lmax[1]]
    yizong=np.sum(channel11)
    channel1[lmax[0],:,lmax[1]]=0
    
    zong1=np.sum(channel1)
    total=zong1+yizong
    if zong1>0:
        sum1=np.sum(channel1,axis=ax).reshape((m,1,sdn))
        bb=sum1/total
        channel=channel1/sum1
        #######################
        channel7=np.log2(channel)*-1*channel
        channel7[np.isnan(channel7)]=0
        sum2=np.sum(channel7,axis=ax).reshape((m,1,sdn))
        ppset=np.zeros((m,1,sdn))
        for zu in range(m):
            for zi in range(sdn):
                if sum1[zu,0,zi]>0:
                    dnum=np.where(channel1[zu,:,zi]>0)
                    pp=max_predictability(sum2[zu,0,zi],len(dnum[0]))
                    ppset[zu,0,zi]=pp
                else:
                    ppset[zu,0,zi]=0
        fpp1=np.sum(bb*ppset)
    else:
        fpp1=0
    fpp=round(fpp1+yizong/total/dd,5)
    return fpp

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


func1=lambda x:time1(x)
func2=lambda x:time2(x)
func8=lambda x:[int(i) for i in x.split('_')]

flag='foursquare_nyc'

name1='../data_preprocessing/base_'+flag+'_v1.txt'
ff=open(name1,encoding='utf-8')

outname1='./'+flag+'_rmp_train.txt'
g1=open(outname1,'w',encoding='utf-8')

for ij in ff.readlines():
    mess=ij.strip().split('#')
    user=int(mess[0])
    print(user)

    binds=mess[-1].split('&')
    sli=[ii for ii in binds if ii=='train']
    trnum=len(sli)
    
    dseq1=mess[1].split('&')
    tseq1=mess[2].split('&')

    ###########
    tseq2=list(map(func1,tseq1))
    tseq3=list(map(func2,tseq1))
    #######
    pdnuml=np.unique(dseq1)
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
    
#        begain=len(dseq11)-1#max(1,len(dseq11)-30)
    ##########################################
    for j in range(1,trnum):
        t2=tseq22[j-1]#@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$
        t3=tseq33[j-1]
        
        d1=dseq11[j-1]
        d2=dseq11[j]
        
        juzhen1[d1,d2,t2]=juzhen1[d1,d2,t2]+1
        juzhen2[d1,d2,t3]=juzhen2[d1,d2,t3]+1
        
    ############## real entropy
    dseq=dseq11[0:trnum]
    mmm=len(np.unique(dseq))
    unc=unc_entropy(dseq)
    ############## fusion conditional entropy
    td77=np.sum(juzhen1,axis=0).transpose()
    od77=np.sum(juzhen1,axis=2)
    dd77=np.sum(np.sum(juzhen1,axis=2),axis=0)
    ff3=shannon(dd77,mmm)#d
    ######### fusion multivaruate sample entropy
    ################ refined fusion conditional entropy final version
    td88=np.sum(juzhen2,axis=0).transpose()
    ke=max(int((j+1)/mmm/8),1)
    r_tod=han_rce_tod(juzhen1,1,ke,pdnum,tdnum2,mmm)
    r_td=han_rce_od(td77,1,ke,tdnum2,pdnum,mmm)#m1,m2,dd
    r_od=han_rce_od(od77,1,ke,pdnum,pdnum,mmm)
    r_d=ff3.split('#')[-1]
                  
    r_tod2=han_rce_tod(juzhen2,1,ke,pdnum,tdnum3,mmm)
    r_td2=han_rce_od(td88,1,ke,tdnum3,pdnum,mmm)#m1,m2,dd
    #####################3
    set77=([mess[0],str(j+1),r_tod,r_td,r_od,r_d,r_tod2,r_td2])
    set77=[str(oo) for oo in set77]
    txt1='$'.join(set77)+'\n'
    g1.write(txt1)
    
g1.close()



    
