#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

class PortfolioAnalysis:
    def __init__(self, file_path, skiprows, name="Portfolio"):
        """
        初始化投资组合分析类
        file_path: CSV文件路径
        skiprows: 需要跳过的行数
        name: 投资组合名称
        """
        self.name = name
        self.df = self._load_data(file_path, skiprows)
        self.T, self.N = self.df.shape
        self.N = self.N - 1  # 减去Date列
        print(f'\n{self.name} Data:')
        print(f'Assets: {self.N}, Periods: {self.T}')

    def _load_data(self, file_path, skiprows):
        """加载并预处理数据"""
        df = pd.read_csv(
            file_path,
            sep=',',
            skiprows=skiprows,
            engine='python',
            header=0,
            na_values=[-99.99, -999]
        )
        return df.dropna()

    def getData(self, start, M):
        """获取指定窗口的收益率数据"""
        return self.df.loc[start:(start+M-1), self.df.columns != 'Date'].astype(float).to_numpy()

    def minVarianceStrategy(self, x):
        """计算最小方差策略的权重"""
        c = np.cov(x, rowvar=False)
        c_inv = np.linalg.inv(c)
        ones = np.ones(len(c_inv))
        numerator = np.matmul(c_inv, ones)
        denominator = np.matmul(ones, numerator)
        return numerator / denominator

    def returns(self, t, w):
        """计算给定时期的收益"""
        x = self.getData(t, 1)
        return np.matmul(x, w)[0]

    def SharpeRatio(self, returns):
        """计算夏普比率"""
        m, s = np.mean(returns), np.std(returns)
        return m/s if s != 0 else 0

    def compute_CEQ(self, r, gamma=1):
        """计算确定性等价收益"""
        mu = np.mean(r)
        var = np.var(r) * 0.01  # 方差乘以0.01
        return (mu - gamma/2 * var) * 0.01  # 最终结果乘以0.01

    def compute_turnover(self, weights):
        """计算换手率"""
        if len(weights) <= 1:
            return 0
        total = 0
        for t in range(len(weights) - 1):
            total += np.sum(np.abs(weights[t+1] - weights[t]))
        return total / (len(weights) - 1)

    def evaluateMinVarianceStrategy(self, M):
        """评估最小方差策略的表现"""
        resInSample, resOutOfSample, weights_list = [], [], []
        max_index = len(self.df) - 1

        try:
            for t in range(self.T-M-2):
                if t+M > max_index or t+M+1 > max_index:
                    continue

                x = self.getData(t, M)
                w = self.minVarianceStrategy(x)
                weights_list.append(w)
                ri = self.returns(t+M, w)
                ro = self.returns(t+M+1, w)
                resInSample.append(ri)
                resOutOfSample.append(ro)

            if not resInSample or not resOutOfSample:
                print(f"Warning: No results generated for window size {M}")
                return None

            CEQ = self.compute_CEQ(resOutOfSample)
            TO = self.compute_turnover(weights_list)
            SRI = self.SharpeRatio(resInSample)
            SRO = self.SharpeRatio(resOutOfSample)

            print(f'\n{self.name} - Window Size {M}:')
            print(f'Sharp ratio: insample = {SRI:.4f}, outpfsample = {SRO:.4f}')
            print(f'CEQ: outofsample = {CEQ:.4f}')
            print(f'Turnover: outofsample = {TO:.4f}')
            return {'SRI': SRI, 'SRO': SRO, 'CEQ': CEQ, 'TO': TO}

        except Exception as e:
            print(f"Error in evaluation: {e}")
            return None

    def naiveStrategy(self):
        """返回相同N个资产的1/N的相同权重"""
        return np.ones(self.N)/self.N

    def evaluateNaiveStrategy(self, M):
        """评估1/N策略的表现"""
        resOutOfSample, weights_list = [], []
        max_index = len(self.df) - 1
        w = self.naiveStrategy()  # 计算1/N权重

        try:
            for t in range(self.T-M-2):
                if t+M > max_index or t+M+1 > max_index:
                    continue

                weights_list.append(w)
                ro = self.returns(t+M+1, w)
                resOutOfSample.append(ro)

            if not resOutOfSample:
                print(f"Warning: No results generated for window size {M}")
                return None

            CEQ = self.compute_CEQ(resOutOfSample)
            TO = self.compute_turnover(weights_list)
            SR = self.SharpeRatio(resOutOfSample)

            print(f'\n{self.name} - Naive Strategy (1/N) - Window Size {M}:')
            print(f'Sharp ratio: outpfsample = {SR:.4f}')
            print(f'CEQ: outofsample = {CEQ:.4f}')
            print(f'Turnover: outofsample = {TO:.4f}')
            return {'SR': SR, 'CEQ': CEQ, 'TO': TO}

        except Exception as e:
            print(f"Error in evaluation: {e}")
            return None

def analyze_portfolios():
    # 定义要分析的投资组合文件
    portfolios = [
        {
            'name': '10 Industry Portfolios数据集',
            'file': '10_Industry_Portfolios阉割版数据.csv',
            'skiprows': 11
        },
        {
            'name': 'FF-1数据集25',
            'file': 'FF-1数据集25_Portfolios_5x5.csv',
            'skiprows': 15
        },
        {
            'name': 'FF-4数据集',
            'file': 'FF-4数据集.csv',
            'skiprows': 3
        },
        {
            'name': 'HML+SMB+US equity market数据集',
            'file': 'HML+SMB+US equity market.csv',
            'skiprows': 3
        },
    ]

    base_path = '/Users/xuhaozhe/Desktop/金融计算'
    window_sizes = range(60, 601, 60)
    results = {}

    for portfolio in portfolios:
        file_path = os.path.join(base_path, portfolio['file'])
        if not os.path.exists(file_path):
            print(f"\nWarning: File not found - {file_path}")
            continue

        print(f"\nAnalyzing {portfolio['name']}...")
        analyzer = PortfolioAnalysis(
            file_path=file_path,
            skiprows=portfolio['skiprows'],
            name=portfolio['name']
        )

        portfolio_results = {'naive': {}, 'min_variance': {}}
        for M in window_sizes:
            # 评估1/N策略
            naive_result = analyzer.evaluateNaiveStrategy(M)
            if naive_result:
                portfolio_results['naive'][M] = naive_result
            
            # 评估最小方差策略
            min_var_result = analyzer.evaluateMinVarianceStrategy(M)
            if min_var_result:
                portfolio_results['min_variance'][M] = min_var_result
        
        results[portfolio['name']] = portfolio_results

    return results

if __name__ == '__main__':
    print("Starting Portfolio Analysis...")
    results = analyze_portfolios()
    
    # 可以在这里添加结果的可视化或额外的分析
    print("\nAnalysis Complete!") 