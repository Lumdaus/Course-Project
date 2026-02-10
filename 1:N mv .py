#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:18:53 2025

@author: xuhaozhe
"""
import pandas as pd
import numpy as np
#jared
# data from Ken French's web site:
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# data set: Industry Portfolios, 10 Industry Portfolios 
fileName = '/Users/xuhaozhe/Desktop/金融计算/FF-4数据集.csv' # change as needed
df = pd.read_csv(
        fileName,
        sep=',',                 # 使用逗号作为分隔符
        skiprows=3,             # 跳过前11行
        engine='python',
        header=0,                # 第一行为列名
        na_values=[-99.99, -999],# 指定缺失值
    )

#df = pd.read_csv(fileName) # read data from csv file
df = df.dropna() # 去掉缺失值
T, N = df.shape # T 是时间周期
N = N - 1 # N是列数
print('Data for %d assests over %d periods' %(N, T)) # shape of data frame
df.head() # display first 5 rows

# 去掉日期列
asset_cols = [col for col in df.columns if col not in ['Date', '交易月份']]

#这个函数从数据框中提取指定时间窗口的资产收益率数据，并转换为NumPy数组。
def getData(df, start, M):
    """返回从start开始的M长度的窗口内N个资产的超额收益
    returns excess returns for the N assets over a window of M periods from start"""
    return df.loc[start:(start+M-1), asset_cols].astype(float).to_numpy() 

#1. 朴素等权重策略
#这是最简单的投资策略，给每个资产分配相同的权重(1/N)。
def naiveStrategy(N):
    """returns equal weights 1/N for N risky assets
    返回相同N个资产的1/N的相同权重""" 
    return np.ones(N)/N

#2. 均值-方差策略

#这实现了马科维茨经典的均值-方差投资组合优化理论，通过最大化风险调整后的回报来计算最优权重。

def optimalWeights(m, c):#均值向量m和协方差矩阵c
    """returns optimal normalized weights (equation 3 in DeMiguel)
    给定均值向量m和协方差矩阵c，返回最优归一化权重
    given mean vector m and covariance matrix c for N risky assets"""
    covI = np.linalg.inv(c) # inverse of covariance协方差矩阵的逆
    w = np.matmul(covI, m) # unnormalized optimal weights未归一的权重
    return w/sum(w) # normalized optimal weights

def meanVarianceStrategy(x):
    """returns optimal normalized weights for N risky assets 
    using Markowitz (1952) mean-variance strategy
    based on sample mean and covariance matrix of observations x
    基于样本均值和协方差矩阵，使用Markowitz均值-方差策略计算最优权重"""
    m = np.mean(x,axis=0) # mean vector
    c = np.cov(x, rowvar=False)
    return optimalWeights(m, c)#返回均值m和协方差c


#这个函数计算给定权重向量在特定时期t的投资组合收益。使用矩阵乘法把收益率和权重相乘。
def returns(df, t, w):
    """Given weight vector w, computes returns in period t
    给定权重向量w，计算t时期的收益"""
    x = getData(df, t, 1)
    return np.matmul(x,w)[0]

def SharpeRatio(r):
    m, s = np.mean(r), np.std(r)
    return m/s

def compute_CEQ(r, gamma=1):
    mu = np.mean(r)
    var = np.var(r)*0.01
    return (mu - gamma/2 * var)*0.01

def compute_turnover(weights):
    total = 0
    for t in range(len(weights) - 1):
        total += np.sum(np.abs(weights[t+1] - weights[t]))
    return total / (len(weights) - 1)

def evaluateNaiveStrategy(df, M):
    N = df.shape[1] - 1
    w = naiveStrategy(N)
    returns_list = []
    weights_list = []
    for t in range(T - M - 2):
        r = returns(df, t+M+1, w)
        returns_list.append(r)
        weights_list.append(w)
    SR = SharpeRatio(returns_list)
    CEQ = compute_CEQ(returns_list)
    TO = compute_turnover(weights_list)
    print('sharpratio(outof sample) for naive strateg=%4.4f,CEQ=%4.4f,Turnover=%4.4f'%(SR,CEQ,TO))
    return SR, CEQ, TO, returns_list

def evaluateMeanVarianceStrategy(df, M):
    returns_list,returnlisti = [],[]
    weights_list = []
    for t in range(T - M - 2):
        x = getData(df, t, M)
        w = meanVarianceStrategy(x)
        r = returns(df, t+M+1, w)
        ri=returns(df, t+M, w)
        returns_list.append(r)
        returnlisti.append(ri)
        weights_list.append(w)
    SRi = SharpeRatio(returnlisti)
    CEQi = compute_CEQ(returnlisti)
    TOi = compute_turnover(returnlisti)
    SR = SharpeRatio(returns_list)
    CEQ = compute_CEQ(returns_list)
    TO = compute_turnover(weights_list)
    print('sharpratio(outof sample) for mv strateg=%4.4f,CEQ=%4.4f,Turnover=%4.4f'%(SR,CEQ,TO))
    print('sharpratio(in sample) for mv strateg=%4.4f,CEQi=%4.4f,Turnoveri=%4.4f'%(SRi,CEQi,TOi))
    return SR, CEQ, TO, returns_list,weights_list

M=60
print('Window size = %d' %M)
SRNS = evaluateNaiveStrategy(df,M)
SRI, SRO,to,returnlist,weight_list= evaluateMeanVarianceStrategy(df,M)
print(len(weight_list))
'''
for M in range(60,601,60):
    print('Window size = %d' %M)
    SRNS = evaluateNaiveStrategy(df,M)
    SRI, SRO,to,returnlist,weight_list= evaluateMeanVarianceStrategy(df,M)
  
    print()
  '''





