#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:58:48 2025

@author: xuhaozhe
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
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


def returns(df, t, w):
    """给定权重向量w，计算t时期的收益
    Given weight vector w, computes returns in period t"""
    x = getData(df, t, 1)
    return np.matmul(x, w)[0]

def SharpeRatio(returns):
    """计算夏普比率
    returns Sharpe ratio given returns""" 
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


# Add the MacKinlay-Pastor strategy implementation
def mackinlayPastorStrategy(x):
    """
    基于MacKinlay-Pastor (2000)模型计算最优权重
    returns optimal weights for N risky assets using the mp strategy where
    Σ = νμμᵀ + σ²Iₙ
    """
    # 计算样本均值向量
    mu = np.mean(x, axis=0)
    
    # 计算样本协方差矩阵
    sample_cov = np.cov(x, rowvar=False)
    N = len(mu)
    
    # 定义目标函数：估计参数ν和σ²，使得Σ = νμμᵀ + σ²Iₙ尽可能接近样本协方差矩阵
    def objective(params):
        nu, sigma_sq = params
        # 构建模型协方差矩阵
        model_cov = nu * np.outer(mu, mu) + sigma_sq * np.eye(N)
        # 计算模型协方差矩阵与样本协方差矩阵之间的Frobenius范数
        diff = model_cov - sample_cov
        return np.sum(diff**2)
    
    # 初始参数估计(正数约束)
    initial_guess = [1.0, 1.0]  # 初始猜测nu和sigma_sq的值
    bounds = [(0.0001, None), (0.0001, None)]  # nu和sigma_sq都应为正数
    
    # 优化参数
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    nu_est, sigma_sq_est = result.x
    
    # 使用估计的参数构建模型协方差矩阵
    model_cov = nu_est * np.outer(mu, mu) + sigma_sq_est * np.eye(N)
    model_cov_inv = np.linalg.inv(model_cov)
    
    # 根据公式(3)计算相对投资组合权重
    numerator = np.matmul(model_cov_inv, mu)
    denominator = np.matmul(np.ones(N), numerator)
    
    weights = numerator / denominator
    return weights

def evaluateMackinlayPastorStrategy(df, M):
    """评估MacKinlay-Pastor策略的表现
    Evaluate the performance of MacKinlay-Pastor strategy"""
    resInSample,resoutofsample,weight_list= [],[],[]
    max_index = len(df) - 1  # 获取最大有效索引
    
    for t in range(len(df)-M-2):
        x = getData(df, t, M)
        # 添加索引检查   
        if t+M > max_index or t+M+1 > max_index:
            continue
            
        w = mackinlayPastorStrategy(x)
        weight_list.append(w)
        ri = returns(df, t+M, w)
        ro=returns(df, t+M+1, w)
        resInSample.append(ri)
        resoutofsample.append(ro)
    CEQ = compute_CEQ(resoutofsample)
    TO = compute_turnover(weight_list)
    SRI,SRO = SharpeRatio(resInSample),SharpeRatio(resoutofsample)
    print('MacKinlay-Pastor sharp ratio: insample = %4.4f,outpfsample=%4.4f' % (SRI,SRO))
    print('MacKinlay-Pastor CEQ:outofsample=%4.6f'%CEQ)
    print('MacKinlay-Pastor turnover:outofsample=%4.4f'%TO)
    return SRO,weight_list


M=60
print('Window size = %d' %M)
SRO,weight_list=evaluateMackinlayPastorStrategy(df, M)
print()
'''
for M in range(60,601,60):
    print('Window size = %d' %M)
    SRO=evaluateMackinlayPastorStrategy(df, M)
    
    print()

'''






