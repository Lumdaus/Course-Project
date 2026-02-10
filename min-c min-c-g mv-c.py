

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:49:31 2025

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

def minVarianceConstrainedStrategy(x):
    """
    基于样本协方差矩阵，使用带卖空限制的最小方差策略计算最优权重 (shortsale-constrained)
    returns optimal weights for N risky assets using global minimum variance portfolio strategy
    with shortsale constraints (non-negative weights)
    """
    N = x.shape[1]  # 资产数量
    c = np.cov(x, rowvar=False)  # 样本协方差矩阵
    
    # 定义目标函数：投资组合方差
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(c, weights))
    
    # 初始权重猜测值
    initial_weights = np.ones(N) / N
    
    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
    ]
    
    # 边界条件（非负权重 - 即卖空限制）
    bounds = tuple((0, None) for _ in range(N))
    
    # 使用scipy的minimize函数求解最优化问题
    result = minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def gMinCStrategy(x, a=0.5/10):
    """
    实现g-min-c策略，这是1/N策略和带卖空限制的最小方差策略的结合
    a ∈ [0, 1/N]，当a=0时等同于min-c策略，当a=1/N时等同于1/N策略
    默认a=1/(2N)，即min-c策略和1/N策略的中间值
    """
    N = x.shape[1]  # 资产数量
    c = np.cov(x, rowvar=False)  # 样本协方差矩阵
    
    # 定义目标函数：投资组合方差
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(c, weights))
    
    # 初始权重猜测值
    initial_weights = np.ones(N) / N
    
    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda w: w - a}  # 每个权重必须大于等于a
    ]
    
    # 边界条件（非负权重 - 即卖空限制）
    bounds = tuple((0, None) for _ in range(N))
    
    # 使用scipy的minimize函数求解最优化问题
    result = minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

# 2.1 卖空限制的均值-方差策略 (Shortsale-Constrained Mean-Variance Strategy)
def shortsaleConstrainedMVStrategy(x):
    """基于样本均值和协方差矩阵，实现卖空限制的均值-方差策略
    returns optimal weights for N risky assets using mean-variance strategy
    with an additional constraint that all weights must be non-negative (no shorting)"""
    m = np.mean(x, axis=0)  # 均值向量
    c = np.cov(x, rowvar=False)  # 协方差矩阵
    n = len(m)  # 资产数量
    
    # 优化目标函数：最大化 μ'w - γ/2 * w'Σw (这里设γ=1)
    def objective(w):
        return -np.dot(m, w) + 0.5 * np.dot(w.T, np.dot(c, w))
    
    # 约束条件：权重之和为1，且所有权重非负（不允许卖空）
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # 权重之和为1
    bounds = tuple((0, None) for _ in range(n))  # 每个权重都非负
    
    # 初始猜测值：等权重
    w0 = np.ones(n) / n
    
    # 使用优化算法求解
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x  # 返回最优权重


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


def evaluateStrategy(strategy_func, M, strategy_name):
    """评估策略的表现
    Evaluate the performance of a given strategy"""
    resInSample ,resoutofsample= [],[]
    max_index = len(df) - 1  # 获取最大有效索引
    for t in range(T-M-2):
        x = getData(df, t, M)
        # 添加索引检查   
        if t+M > max_index or t+M+1 > max_index:
            continue
        w = strategy_func(x)
        ri = returns(df, t+M, w)
        ro=returns(df, t+M+1, w)
        resInSample.append(ri)
        resoutofsample.append(ro)
    
    SRI = SharpeRatio(resInSample)
    SRO=SharpeRatio(resoutofsample)
    print(f'{strategy_name}策略夏普比率: insample = %4.4f,outpfsample=%4.4f' % (SRI,SRO))
    return SRO


for M in range(60,601,60):
    print(f'\n窗口大小 = {M}')
    
    # 评估带卖空限制的最小方差策略
    sr_min_var_constrained = evaluateStrategy(minVarianceConstrainedStrategy, M, "min-c")
    
    # 评估g-min-c策略
    sr_gminc = evaluateStrategy(gMinCStrategy, M, "g-min-c")

    sr_mv_constrained=evaluateStrategy(shortsaleConstrainedMVStrategy, M, 'mv-c')