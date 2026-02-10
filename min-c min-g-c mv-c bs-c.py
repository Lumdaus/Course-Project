#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:18:39 2025

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

def minVarianceStrategy(x):
    """
    Original minimum variance strategy without constraints
    Returns optimal weights for N risky assets
    """
    c = np.cov(x, rowvar=False)  # Sample covariance matrix
    c_inv = np.linalg.inv(c)      # Inverse of covariance matrix
    ones = np.ones(len(c_inv))    # Vector of ones
    
    # Global minimum variance portfolio weights formula
    # w = (C^-1 * 1) / (1^T * C^-1 * 1)
    numerator = np.matmul(c_inv, ones)
    denominator = np.matmul(ones, numerator)
    
    weights = numerator / denominator
    return weights

def minVarianceStrategyConstrained(x):
    """
    Minimum variance constrained (min-c) strategy
    Adds non-negativity constraints to prevent short-selling
    """
    n = x.shape[1]  # Number of assets
    c = np.cov(x, rowvar=False)  # Sample covariance matrix
    
    # Define the objective function to minimize (portfolio variance)
    def objective(w):
        return w @ c @ w
    
    # Define constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Define bounds: weights must be non-negative (no short selling)
    bounds = tuple((0, None) for _ in range(n))
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Perform optimization
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        # If optimization fails, return equal weights
        return np.ones(n) / n

def meanVarianceConstrainedStrategy(x, risk_aversion=1.0):
    """
    Mean-variance constrained (mv-c) strategy
    Incorporates both expected returns and variance with short-sale constraints
    """
    n = x.shape[1]  # Number of assets
    c = np.cov(x, rowvar=False)  # Sample covariance matrix
    mu = np.mean(x, axis=0)      # Sample mean returns
    
    # Define the objective function to maximize (mean - risk_aversion * variance)
    # Since minimize is used, we negate the objective
    def objective(w):
        return -(mu @ w - (risk_aversion/2) * (w @ c @ w))
    
    # Define constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Define bounds: weights must be non-negative (no short selling)
    bounds = tuple((0, None) for _ in range(n))
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Perform optimization
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        # If optimization fails, return equal weights
        return np.ones(n) / n

def bayesSteinConstrainedStrategy(x, risk_aversion=1.0):
    """
    Bayes-Stein constrained (bs-c) strategy
    Applies Bayes-Stein shrinkage to expected returns with short-sale constraints
    """
    n = x.shape[1]  # Number of assets
    T = x.shape[0]  # Number of observations
    
    c = np.cov(x, rowvar=False)  # Sample covariance matrix
    c_inv = np.linalg.inv(c)     # Inverse of covariance matrix
    
    mu = np.mean(x, axis=0)      # Sample mean returns
    mu_g = np.mean(mu)           # Grand mean
    ones = np.ones(n)            # Vector of ones
    
    # Calculate shrinkage intensity
    # Formula from Jorion (1986)
    q = (n + 2) / (
        (T + 2) * 
        ((mu - mu_g * ones) @ c_inv @ (mu - mu_g * ones))
    )
    
    # Shrink the sample means toward the grand mean
    mu_bs = (1 - q) * mu + q * mu_g * ones
    
    # Define the objective function to maximize (mean - risk_aversion * variance)
    # Since minimize is used, we negate the objective
    def objective(w):
        return -(mu_bs @ w - (risk_aversion/2) * (w @ c @ w))
    
    # Define constraints: sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Define bounds: weights must be non-negative (no short selling)
    bounds = tuple((0, None) for _ in range(n))
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Perform optimization
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        # If optimization fails, return equal weights
        return np.ones(n) / n

def gMinVarianceConstrainedStrategy(x, a=0.5):
    """
    Generalized minimum-variance constrained (g-min-c) strategy
    Combination of 1/N policy and constrained minimum-variance strategy
    w >= a * 1/N, with a in [0, 1/N]
    """
    n = x.shape[1]  # Number of assets
    c = np.cov(x, rowvar=False)  # Sample covariance matrix
    equal_weight = 1/n
    min_weight = a * equal_weight
    
    # Define the objective function to minimize (portfolio variance)
    def objective(w):
        return w @ c @ w
    
    # Define constraints
    # 1. Sum of weights = 1
    # 2. Each weight >= a*(1/N)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    
    # Add lower bound constraints for each weight
    for i in range(n):
        constraints.append(
            {'type': 'ineq', 'fun': lambda w, i=i: w[i] - min_weight}
        )
    
    # Define bounds: weights must be non-negative (already covered by the constraints)
    bounds = tuple((0, None) for _ in range(n))
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Perform optimization
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        # If optimization fails, return equal weights
        return np.ones(n) / n

# -------------- Evaluation Functions --------------

def evaluateStrategy(strategy_func, strategy_name, M, **kwargs):
    """Evaluate the performance of a strategy"""
    resInSample, resOutOfSample,weight_list= [], [],[]
    max_index = len(df) - 1  # Get maximum valid index
    
    for t in range(T-M-2):
        x = getData(df, t, M)
        # Check if indices are valid
        if t+M > max_index or t+M+1 > max_index:
            continue
            
        # Apply the strategy
        w = strategy_func(x, **kwargs)
        weight_list.append(w)
        ri = returns(df, t+M, w)
        ro=returns(df, t+M+1, w)
        
        resInSample.append(ri)
        resOutOfSample.append(ro)
    CEQ = compute_CEQ(resOutOfSample)
    TO = compute_turnover(weight_list)
    SRI,SRO= SharpeRatio(resInSample),SharpeRatio(resOutOfSample)
    print(f'{strategy_name} Sharpe Ratio: In-Sample = {SRI:.4f},outfofsample={SRO:.4f}')
    print(f'{strategy_name} CEQ: outfofsample={CEQ:.6f}')
    print(f'{strategy_name} Turnover: outfofsample={TO:.4f}')
    return SRO
M=60
print('Window size = %d' %M)
print(f'\nWindow size = {M}')
 # Evaluate minimum variance constrained strategy
SRO_minvarc = evaluateStrategy(minVarianceStrategyConstrained, "Minimum Variance Constrained (min-c)", M)
 
 # Evaluate mean-variance constrained strategy
SRO_mvc = evaluateStrategy(meanVarianceConstrainedStrategy, "Mean-Variance Constrained (mv-c)", M, risk_aversion=1.0)
 
 # Evaluate Bayes-Stein constrained strategy
SRO_bsc = evaluateStrategy(bayesSteinConstrainedStrategy, "Bayes-Stein Constrained (bs-c)", M, risk_aversion=1.0)
 
 # Evaluate generalized minimum variance constrained strategy
 # Using a = 0.5 * (1/N) as mentioned in the paper
SRO_gminc = evaluateStrategy(gMinVarianceConstrainedStrategy, "Generalized Min-Var Constrained (g-min-c)", M, a=0.5)



'''
# -------------- Main Execution --------------

if __name__ == "__main__":
    print("\nEvaluating different portfolio strategies...")
    window_sizes = range(60, 601, 60)
    
    for M in window_sizes:
        print(f'\nWindow size = {M}')
        # Evaluate minimum variance constrained strategy
        SRO_minvarc = evaluateStrategy(minVarianceStrategyConstrained, "Minimum Variance Constrained (min-c)", M)
        
        # Evaluate mean-variance constrained strategy
        SRO_mvc = evaluateStrategy(meanVarianceConstrainedStrategy, "Mean-Variance Constrained (mv-c)", M, risk_aversion=1.0)
        
        # Evaluate Bayes-Stein constrained strategy
        SRO_bsc = evaluateStrategy(bayesSteinConstrainedStrategy, "Bayes-Stein Constrained (bs-c)", M, risk_aversion=1.0)
        
        # Evaluate generalized minimum variance constrained strategy
        # Using a = 0.5 * (1/N) as mentioned in the paper
        SRO_gminc = evaluateStrategy(gMinVarianceConstrainedStrategy, "Generalized Min-Var Constrained (g-min-c)", M, a=0.5)
        
        print("\nComparison of Sharpe Ratios:")
        print(f"Min-Var Constrained: {SRO_minvarc:.4f}")
        print(f"Mean-Var Constrained: {SRO_mvc:.4f}")
        print(f"Bayes-Stein Constrained: {SRO_bsc:.4f}")
        print(f"Generalized Min-Var Constrained: {SRO_gminc:.4f}")
        
        
        '''