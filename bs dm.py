#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:08:14 2025

@author: xuhaozhe
"""

import pandas as pd
import numpy as np
from scipy import stats
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

def bayesSteinStrategy(x, M):
    
    """
    基于Bayes-Stein收缩估计的投资组合策略
    Implementation based on Jorion (1985, 1986)
    
    参数:
    x: 样本资产收益矩阵
    M: 样本窗口大小
    
    返回:
    bsw: Bayes-Stein优化权重
    """
    N = x.shape[1]  # 资产数量
    
    # 1. 计算样本均值和协方差矩阵
    mu = np.mean(x, axis=0)  # 样本均值
    sigma = np.cov(x, rowvar=False)  # 样本协方差矩阵
    sigma_inv = np.linalg.inv(sigma)  # 协方差矩阵的逆
    
    # 2. 计算最小方差投资组合
    ones = np.ones(N)
    w_min = np.matmul(sigma_inv, ones)
    w_min = w_min / np.sum(w_min)  # 归一化最小方差组合权重
    
    # 3. 计算最小方差投资组合的均值作为"grand mean"
    mu_min = np.matmul(mu, w_min)
    
    # 4. 计算收缩强度参数 phi
    # 使用公式 (5) 中的计算方法
    diff = mu - mu_min * ones
    phi_numerator = N + 2
    phi_denominator = (N + 2) + M * np.matmul(np.matmul(diff, sigma_inv), diff)
    phi = phi_numerator / phi_denominator
    
    # 确保phi在[0,1]范围内
    phi = np.clip(phi, 0, 1)
    
    # 5. 计算收缩后的均值估计 (公式4)
    mu_bs = (1 - phi) * mu + phi * mu_min * ones
    
    # 6. 使用收缩后的均值构建有效投资组合
    # 注意：这里我们使用最小方差组合(而不是均值-方差优化)，简化实现
    # 在实际应用中，可能需要求解二次规划问题
    
    # 为简化实现，我们直接使用收缩后的均值和原始协方差矩阵
    # 构建一个投资组合权重向量
    # 这是简化版本，实际可能需要均值-方差优化
    mu_bs_norm = mu_bs / np.sum(mu_bs)  # 简单归一化处理
    
    # 应用投资组合优化理论，使用收缩均值和样本协方差
    # 在实际应用中，Jorion还建议对协方差矩阵进行贝叶斯调整
    # 这里我们仅关注均值收缩
    
    # 构建简化的投资组合权重 (这是简化版本)
    # 实际上，应该解决完整的均值-方差优化问题
    bsw = np.matmul(sigma_inv, mu_bs)
    bsw = bsw / np.sum(bsw)  # 归一化权重
    
    return bsw

def evaluateBayesSteinStrategy(M):
    """评估Bayes-Stein策略的表现"""
    resInSample ,resoutofsample,weights_list= [],[],[]
    max_index = len(df) - 1
    for t in range(T-M-2):
        if t+M > max_index or t+M+1 > max_index:
            continue
        x = getData(df, t, M)
        w = bayesSteinStrategy(x, M)
       
        ri = returns(df, t+M, w)
        ro=returns(df, t, w)
        resInSample.append(ri)
        resoutofsample.append(ro)
        weights_list.append(w)
    CEQ = compute_CEQ(resoutofsample)
    TO = compute_turnover(weights_list)
    SRI = SharpeRatio(resInSample)
    SRO=SharpeRatio(resoutofsample)
    SRI = SharpeRatio(resInSample)
    print('Bayes-Stein sharp ratio: insample = %4.4f,outpfsample=%4.4f,outsample CEQ=%4.6f, out of sample Turnover=%4.4f' % (SRI,SRO,CEQ,TO))
    return SRI

# ----- 新增: Data-and-Model 策略实现 -----

def dataAndModelStrategy(x, M, model='CAPM', alpha_sd=0.01):
    """
    基于"Data-and-Model"方法的投资组合策略，
    根据Pástor (2000)和Pástor and Stambaugh (2000)的方法
    
    参数:
    x: 样本资产收益矩阵
    M: 样本窗口大小
    model: 使用的资产定价模型 ('CAPM', 'FF3', 'Carhart4')
    alpha_sd: 定价误差的先验标准差 (默认每年1%)
    
    返回:
    dmw: Data-and-Model优化权重
    """
    N = x.shape[1]  # 资产数量
    
    # 1. 计算样本均值和协方差矩阵
    mu = np.mean(x, axis=0)
    sigma = np.cov(x, rowvar=False)
    sigma_inv = np.linalg.inv(sigma)
    
    # 2. 假设使用CAPM模型 (简化版本)
    # 在实际应用中，需要实现CAPM、FF3或Carhart4因子模型的估计
    # 这里我们简化处理，用市场投资组合作为因子
    
    # 2.1 创建一个简单的市场投资组合 (等权)
    market_weights = np.ones(N) / N
    market_return = np.matmul(x, market_weights)
    
    # 2.2 估计CAPM贝塔系数 (针对每个资产)
    betas = np.zeros(N)
    for i in range(N):
        # 简单回归估计贝塔系数
        beta, _, _, _ = np.linalg.lstsq(
            market_return.reshape(-1, 1), 
            x[:, i], 
            rcond=None
        )
        betas[i] = beta[0]
    
    # 2.3 估计模型预测的期望收益
    market_mean = np.mean(market_return)
    mu_model = betas * market_mean  # 根据CAPM的预期收益
    
    # 3. 计算定价误差 (alpha)
    alpha = mu - mu_model
    
    # 4. 计算先验的精度矩阵
    # 假设alpha的先验分布是均值为0，标准差为alpha_sd的正态分布
    # 转换为月度标准差 (假设alpha_sd是年度值)
    monthly_alpha_sd = alpha_sd / np.sqrt(12)
    alpha_precision = 1 / (monthly_alpha_sd ** 2)
    
    # 5. 结合数据和模型计算后验均值
    # 根据贝叶斯理论，后验均值是先验均值和样本均值的加权平均
    # 权重与各自精度成正比
    
    # 样本精度 (简化: 假设资产间独立)
    data_precision = M / np.diag(sigma)  # 简化处理
    
    # 计算后验估计
    posterior_precision = alpha_precision + data_precision
    posterior_mu = (alpha_precision * mu_model + data_precision * mu) / posterior_precision
    
    # 6. 构建投资组合
    # 使用后验均值和样本协方差矩阵
    dmw = np.matmul(sigma_inv, posterior_mu)
    dmw = dmw / np.sum(dmw)  # 归一化权重
    
    return dmw

def evaluateDataAndModelStrategy(M):
    """评估Data-and-Model策略的表现"""
    resInSample,resoutofsample,weights_list= [],[],[]
    max_index = len(df) - 1
    for t in range(T-M-2):
        if t+M > max_index or t+M+1 > max_index:
            continue
        x = getData(df, t, M)
        w = dataAndModelStrategy(x, M)
        ri = returns(df, t+M, w)
        ro=returns(df, t+M+1, w)
        resInSample.append(ri)
        resoutofsample.append(ro)
        weights_list.append(w)
    CEQ = compute_CEQ(resoutofsample)
    TO = compute_turnover(weights_list)
    SRI = SharpeRatio(resInSample)
    SRO=SharpeRatio(resoutofsample)
    
    print('Data-and-Model sharp ratio: insample = %4.4f,outpfsample=%4.4f,outsample CEQ=%4.6f, out of sample Turnover=%4.4f' % (SRI,SRO,CEQ,TO))
    return SRO

M=60
print('Window size = %d' %M)
SRIofbs=evaluateBayesSteinStrategy(M)

SRIofdm= evaluateDataAndModelStrategy(M)
print()


'''
for M in range(60,601,60):
    print('Window size = %d' %M)
    SRIofbs=evaluateBayesSteinStrategy(M)
   
    SRIofdm= evaluateDataAndModelStrategy(M)
    print()
# ----- 所有策略评估 -----
'''
'''
def evaluateAllStrategies(M_values):
    """评估所有策略在不同窗口大小下的表现"""
    results = {'Window': [], 'Min Variance': [], 'Bayes-Stein': [], 'Data-and-Model': []}
    
    for M in M_values:
        print('-' * 50)
        print(f'Window size = {M}')
        
        # 记录窗口大小
        results['Window'].append(M)
        
        
        # Bayes-Stein策略
        bs_sr = evaluateBayesSteinStrategy(M)
        results['Bayes-Stein'].append(bs_sr)
        
        # Data-and-Model策略
        dm_sr = evaluateDataAndModelStrategy(M)
        results['Data-and-Model'].append(dm_sr)
        
        print()
    
    return pd.DataFrame(results)

# 主程序
if __name__ == "__main__":
    # 测试不同窗口大小
    M_values = range(60, 601, 60)
    
    # 评估所有策略
    results_df = evaluateAllStrategies(M_values)
    
    # 打印结果表格
    print("\n策略表现比较 (夏普比率):")
    print(results_df)
    
    # 打印最佳窗口大小(按不同策略)
    print("\n不同策略的最佳窗口大小:")
    for strategy in ['Min Variance', 'Bayes-Stein', 'Data-and-Model']:
        best_idx = results_df[strategy].idxmax()
        best_window = results_df.loc[best_idx, 'Window']
        best_sr = results_df.loc[best_idx, strategy]
        print(f"{strategy}: 窗口大小 = {best_window}, 夏普比率 = {best_sr:.4f}")
'''