#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:25:05 2025

@author: luy
"""

import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def handle_time(x):
    mm=int(x.split('-')[1])
    ww=datetime.strptime(x,"%Y-%m-%d %H:%M:%S").weekday()
    hh=int(x.split(' ')[-1].split(':')[0])
    if hh>=6 and hh<10:
        shiduan=0
    elif hh>=10 and hh<16:
        shiduan=1
    elif hh>=16 and hh<20:
        shiduan=2
    else:
        shiduan=3
    
    if ww>4:
        shiduan=shiduan+4
    set1=([mm,shiduan,ww,hh])
    set1=[str(x) for x in set1]
    txt='_'.join(set1)
    return txt


func1=lambda x:handle_time(x)
func2=lambda x:x.str.cat(sep='&')

flag='foursquare_nyc'
name1='./data/'+flag+'.txt'

df1=pd.read_csv(name1,header=None,sep='#')

df1[100]=df1[7].apply(func1)

fd=df1.iloc[:,[0,1,-1,-2]]

fd[1]=fd[1].map(str)
fd[100]=fd[100].map(str)
fd[8]=fd[8].map(str)

dfg1=fd.groupby(0)[1].apply(func2)
dfg2=fd.groupby(0)[100].apply(func2)
dfg3=fd.groupby(0)[8].apply(func2)

dfgz=pd.concat([dfg1,dfg2,dfg3],axis=1)

outname='./base_'+flag+'_v1.txt'
dfgz.to_csv(outname,sep='#',index=True,header=None,encoding='utf-8')
