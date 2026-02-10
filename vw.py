#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 08:43:00 2025

@author: xuhaozhe
"""

import pandas as pd
import numpy as np
# data from Ken French's web site:
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# data set: Industry Portfolios, 10 Industry Portfolios 
fileName = '/Users/xuhaozhe/Desktop/金融计算/10_Industry_Portfolios阉割版数据.csv' # change as needed
df = pd.read_csv(
        fileName,
        sep=',',                 # 使用逗号作为分隔符
        skiprows=11,             # 跳过前11行
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


#这个函数从数据框中提取指定时间窗口的资产收益率数据，并转换为NumPy数组。
def getData(df, start, M):
    """返回从start开始的M长度的窗口内N个资产的超额收益
    returns excess returns for the N assets over a window of M periods from start"""
    return df.loc[start:(start+M-1),df.columns != 'Date'].astype(float).to_numpy() 



# 3. 价值加权策略 (Value-Weighted)
def valueWeightedStrategy(x):
    """基于市场价值实现价值加权策略
    returns value-weighted portfolio weights based on the average returns
    which simulates market capitalization in this simplified approach"""
    # 在这个简化模型中，我们使用平均收益作为市值的代理
    # 实际应用中应使用真实的市值数据
    market_values = np.mean(x, axis=0)
    
    # 确保所有权重为正（如果出现负数，转为绝对值）
    market_values = np.abs(market_values)
    
    # 归一化权重，使其总和为1
    return market_values / np.sum(market_values)


def returns(df, t, w):
    """给定权重向量w，计算t时期的收益
    Given weight vector w, computes returns in period t"""
    x = getData(df, t, 1)
    return np.matmul(x, w)[0]


# 计算夏普比率
def SharpeRatio(returns):
    """returns Sharpe ratio given returns""" 
    m, s = np.mean(returns), np.std(returns)
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

# 评估价值加权策略
def evaluateValueWeightedStrategy(M):
    resInSample, resOutOfSample,weight_list = [], [],[]
    max_index = len(df) - 1  # 获取最大有效索引
    for t in range(T-M-2):
        x = getData(df, t, M)
        # 添加索引检查
        if t+M > max_index or t+M+1 > max_index:
            continue
        w = valueWeightedStrategy(x)
        weight_list.append(w)
        ri = returns(df, t+M, w)
        ro=returns(df, t+M+1, w)
        resInSample.append(ri)
        resOutOfSample.append(ro)
    CEQ = compute_CEQ(resOutOfSample)
    TO = compute_turnover(weight_list)    
    SRI,SRO = SharpeRatio(resInSample),SharpeRatio(resOutOfSample)
    print('Sharpe Ratio for Value Weighted Strategy: insample = %4.4f,outpfsample=%4.4f' % (SRI,SRO))
    print('CEQ for Value Weighted Strategy:outofsample=%4.6f'%CEQ)
    print('turnover for Value Weighted Strategy:outofsample=%4.4f'%TO)
    return SRI

M=60
print('\nWindow size = %d' %M)
SRI_VW = evaluateValueWeightedStrategy(M)
print()
'''
# 运行评估
print("评估不同窗口大小下的投资策略表现:")
for M in range(60, 601, 60):
    print('\nWindow size = %d' %M)
    SRI_VW = evaluateValueWeightedStrategy(M)
    print()
'''
















