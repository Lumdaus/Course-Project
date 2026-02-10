#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:29:52 2025

@author: xuhaozhe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def compute_CEQ(r, gamma=1):
    mu = np.mean(r)
    var = np.var(r)*0.01
    return (mu - gamma/2 * var)*0.01

def compute_turnover(weights):
    total = 0
    for t in range(len(weights) - 1):
        total += np.sum(np.abs(weights[t+1] - weights[t]))
    return total / (len(weights) - 1)

# 4. MV-MIN策略 (Mean-Variance and Minimum-Variance mixture)
def mvMinStrategy(x, c=0.5, d=0.5):
    """
    实现Kan和Zhou(2007)的三基金（mv-min）投资组合策略
    结合了均值方差策略和最小方差策略
    
    参数:
    x: 收益率数据
    c, d: 混合参数，默认为各占50%
    
    返回: 混合策略的最优权重
    """
    # 计算均值向量和协方差矩阵
    mu = np.mean(x, axis=0)  # 均值向量
    sigma = np.cov(x, rowvar=False)  # 协方差矩阵
    sigma_inv = np.linalg.inv(sigma)  # 协方差矩阵的逆
    
    # 计算未归一化的权重 (公式10)
    ones = np.ones(len(mu))
    w_unnorm = (1/1) * (c * np.matmul(sigma_inv, mu) + d * np.matmul(sigma_inv, ones))
    
    # 归一化权重
    w_norm = w_unnorm / np.matmul(ones, w_unnorm)
    
    return w_norm

# 5. EW-MIN策略 (Equally-Weighted and Minimum-Variance mixture)
def ewMinStrategy(x, c=0.5, d=0.5):
    """
    实现等权重和最小方差的混合投资组合策略
    
    参数:
    x: 收益率数据
    c, d: 混合参数，默认为各占50%
    
    返回: 混合策略的最优权重
    """
    N = x.shape[1]  # 资产数量
    
    # 计算协方差矩阵及其逆
    sigma = np.cov(x, rowvar=False)
    sigma_inv = np.linalg.inv(sigma)
    
    # 等权重向量 (1/N)
    ones = np.ones(N)
    w_ew = ones / N
    
    # 计算未归一化的权重 (公式11)
    w_unnorm = c * w_ew + d * np.matmul(sigma_inv, ones)
    
    # 归一化权重确保权重和为1
    w_norm = w_unnorm / np.sum(w_unnorm)
    
    return w_norm

# 6. 优化求解C和D参数的函数
def optimize_mixture_params(x, strategy='mv-min'):
    """
    优化混合策略的c和d参数，使期望效用最大化
    
    参数:
    x: 收益率数据
    strategy: 'mv-min' 或 'ew-min'
    
    返回: 最优的c和d参数
    """
    # 风险厌恶系数
    gamma = 1#随后更改风险厌恶系数进行参数调整
    
    # 目标函数：负的期望效用 (mean - gamma/2 * variance)
    def neg_expected_utility(params):
        c, d = params
        
        if strategy == 'mv-min':
            w = mvMinStrategy(x, c, d)
        else:  # ew-min
            w = ewMinStrategy(x, c, d)
            
        # 计算预期收益和方差
        mu = np.mean(x, axis=0)
        sigma = np.cov(x, rowvar=False)
        
        expected_return = np.dot(w, mu)
        variance = np.dot(w, np.dot(sigma, w))
        
        # 期望效用 (负值因为我们要最小化)
        utility = -(expected_return - (gamma/2) * variance)
        
        return utility
    
    # 约束条件: c + d = 1
    def constraint(params):
        return params[0] + params[1] - 1
    
    # 边界和初始猜测
    bounds = [(0, 1), (0, 1)]
    constraints = {'type': 'eq', 'fun': constraint}
    initial_guess = [0.5, 0.5]
    
    # 最小化负期望效用
    result = minimize(neg_expected_utility, initial_guess, 
                      method='SLSQP', bounds=bounds, 
                      constraints=constraints)
    
    if result.success:
        return result.x
    else:
        print("优化失败，使用默认值c=0.5, d=0.5")
        return np.array([0.5, 0.5])


# 计算给定权重向量在特定时期t的投资组合收益
def returns(df, t, w):
    """给定权重向量w，计算t时期的收益"""
    x = getData(df, t, 1)
    return np.matmul(x, w)[0]

# 计算夏普比率
def SharpeRatio(returns):
    """给定收益，计算夏普比率""" 
    m, s = np.mean(returns), np.std(returns)
    return m/s


# 评估MV-MIN策略
def evaluateMvMinStrategy(M):
    """评估MV-MIN策略"""
    resInSample, resOutOfSample ,weight_list= [], [],[]
    max_index = len(df) - 1  # 获取最大有效索引
    
    # 记录最优参数
    optimal_params = []
    
    for t in range(T-M-2):
        if t+M > max_index or t+M+1 > max_index:
            continue
        x = getData(df, t, M)
        
        # 优化c和d参数
        c, d = optimize_mixture_params(x, 'mv-min')
        optimal_params.append((c, d))
        
        # 计算权重和收益
        w = mvMinStrategy(x, c, d)
        weight_list.append(w)
        ri = returns(df, t+M, w)
        ro = returns(df, t+M+1, w)
        
        resInSample.append(ri)
        resOutOfSample.append(ro)
    CEQ = compute_CEQ(resOutOfSample)
    TO = compute_turnover(weight_list)
    SRI, SRO = SharpeRatio(resInSample), SharpeRatio(resOutOfSample)
    
    # 计算平均最优参数
    avg_c, avg_d = np.mean([p[0] for p in optimal_params]), np.mean([p[1] for p in optimal_params])
    
    print('MV-MIN策略夏普比率: insample = %4.4f,outpfsample=%4.4f' % (SRI,SRO))
    print('MV-MIN策略CQE: outpfsample=%4.6f' % CEQ)
    print('MV-MIN策略Turnover: outpfsample=%4.4f' % TO)
    print('MV-MIN平均参数: c = %4.4f, d = %4.4f' % (avg_c, avg_d))
    
    return SRI, SRO, avg_c, avg_d

# 评估EW-MIN策略
def evaluateEwMinStrategy(M):
    """评估EW-MIN策略"""
    resInSample, resOutOfSample,weight_list = [], [],[]
    max_index = len(df) - 1  # 获取最大有效索引
    
    # 记录最优参数
    optimal_params = []
    
    for t in range(T-M-2):
        if t+M > max_index or t+M+1 > max_index:
            continue
        x = getData(df, t, M)
        
        # 优化c和d参数
        c, d = optimize_mixture_params(x, 'ew-min')
        optimal_params.append((c, d))
        
        # 计算权重和收益
        w = ewMinStrategy(x, c, d)
        weight_list.append(w)
        ri = returns(df, t+M, w)
        ro = returns(df, t+M+1, w)
        
        resInSample.append(ri)
        resOutOfSample.append(ro)
    CEQ = compute_CEQ(resOutOfSample)
    TO = compute_turnover(weight_list)
    
    SRI, SRO = SharpeRatio(resInSample), SharpeRatio(resOutOfSample)
    
    # 计算平均最优参数
    avg_c, avg_d = np.mean([p[0] for p in optimal_params]), np.mean([p[1] for p in optimal_params])
    
    print('EW-MIN策略夏普比率: insample = %4.4f,outpfsample=%4.4f' % (SRI,SRO))
    print('EW-MIN策略CQE: outpfsample=%4.6f' % CEQ)
    print('EW-MIN策略Turnover: outpfsample=%4.4f' % TO)
    print('EW-MIN平均参数: c = %4.4f, d = %4.4f' % (avg_c, avg_d))
    
    return SRI, SRO, avg_c, avg_d

M=60
print('\n窗口大小 = %d' % M)
      
        
        # 评估MV-MIN策略
SRI_mvmin, SRO_mvmin, c_mvmin, d_mvmin = evaluateMvMinStrategy(M)
       
        # 评估EW-MIN策略
SRI_ewmin, SRO_ewmin, c_ewmin, d_ewmin = evaluateEwMinStrategy(M)

'''
def main():
    # 存储结果的数据结构
  
    # 测试不同窗口大小
    for M in range(60, 601, 60):
        print('\n窗口大小 = %d' % M)
      
        
        # 评估MV-MIN策略
        SRI_mvmin, SRO_mvmin, c_mvmin, d_mvmin = evaluateMvMinStrategy(M)
       
        # 评估EW-MIN策略
        SRI_ewmin, SRO_ewmin, c_ewmin, d_ewmin = evaluateEwMinStrategy(M)
  

if __name__ == "__main__":
    main()




'''






























































