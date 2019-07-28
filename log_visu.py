# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:55:55 2019

@author: Yhq
"""

import matplotlib.pyplot as plt
import numpy as np
import re

def train_log():
    filename = 'train_log/train.log'
    lossname = ['g_loss', 'd_loss']
    epoch=24
    
    file = open(filename, 'r')
    file = file.readlines()
    
    patterns = []
    for i in range (len(lossname)):
        patterns.append(re.compile(lossname[i] + r': \d+.\d+'))
        
    tot = [[] for i in range (2)]
        
    for line in file:
        for i in range (2):
            s = patterns[i].findall(line)
            if (len(s) == 0):
                break
            tot[i].append(float(s[0].split(':')[1]))


    x = np.linspace(0,epoch,len(tot[0]))

    for i in range (2):
        plt.figure(figsize=(12,8))
        plt.plot(x, tot[i], label=lossname[i])
        plt.legend()
        plt.savefig('train_log/log'+str(i)+'.jpg')


###############################################################################

def train_e_log():
    filename = 'train_log/train-e.log'
    lossname = ['loss']
    step = 48000
    
    file = open(filename, 'r')
    file = file.readlines()
    
    patterns = []
    for i in range (len(lossname)):
        patterns.append(re.compile(lossname[i] + r': \d+.\d+'))
        
    tot = [[] for i in range (1)]
        
    for line in file:
        for i in range (1):
            s = patterns[i].findall(line)
            if (len(s) == 0):
                break
            tot[i].append(float(s[0].split(':')[1]))


    x = np.linspace(0,step,len(tot[0]))

    for i in range (1):
        plt.figure(figsize=(12,8))
        plt.plot(x, tot[i], label=lossname[i])
        plt.legend()
        plt.savefig('train_log/log-e'+str(i)+'.jpg')
    
train_e_log()