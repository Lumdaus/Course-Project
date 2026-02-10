#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.optimize import minimize
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

    # 策略实现
    def naiveStrategy(self):
        """1/N策略"""
        return np.ones(self.N)/self.N

    def minVarianceStrategy(self, x):
        """最小方差策略（无约束）"""
        c = np.cov(x, rowvar=False)
        c_inv = np.linalg.inv(c)
        ones = np.ones(len(c_inv))
        numerator = np.matmul(c_inv, ones)
        denominator = np.matmul(ones, numerator)
        return numerator / denominator

    def minVarianceStrategyConstrained(self, x):
        """最小方差策略（有约束）"""
        n = x.shape[1]
        c = np.cov(x, rowvar=False)
        
        def objective(w):
            return w @ c @ w
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, None) for _ in range(n))
        w0 = np.ones(n) / n
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def meanVarianceConstrainedStrategy(self, x, risk_aversion=1.0):
        """均值-方差策略（有约束）"""
        n = x.shape[1]
        c = np.cov(x, rowvar=False)
        mu = np.mean(x, axis=0)
        
        def objective(w):
            return -(mu @ w - (risk_aversion/2) * (w @ c @ w))
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, None) for _ in range(n))
        w0 = np.ones(n) / n
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def bayesSteinConstrainedStrategy(self, x, risk_aversion=1.0):
        """贝叶斯-斯坦因策略（有约束）"""
        n = x.shape[1]
        T = x.shape[0]
        
        c = np.cov(x, rowvar=False)
        c_inv = np.linalg.inv(c)
        mu = np.mean(x, axis=0)
        mu_g = np.mean(mu)
        ones = np.ones(n)
        
        q = (n + 2) / ((T + 2) * ((mu - mu_g * ones) @ c_inv @ (mu - mu_g * ones)))
        mu_bs = (1 - q) * mu + q * mu_g * ones
        
        def objective(w):
            return -(mu_bs @ w - (risk_aversion/2) * (w @ c @ w))
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, None) for _ in range(n))
        w0 = np.ones(n) / n
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def gMinVarianceConstrainedStrategy(self, x, a=0.5):
        """广义最小方差策略（有约束）"""
        n = x.shape[1]
        c = np.cov(x, rowvar=False)
        equal_weight = 1/n
        min_weight = a * equal_weight
        
        def objective(w):
            return w @ c @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        for i in range(n):
            constraints.append(
                {'type': 'ineq', 'fun': lambda w, i=i: w[i] - min_weight}
            )
        
        bounds = tuple((0, None) for _ in range(n))
        w0 = np.ones(n) / n
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def evaluateStrategy(self, strategy_func, strategy_name, M, **kwargs):
        """评估策略表现"""
        resInSample, resOutOfSample, weights_list = [], [], []
        max_index = len(self.df) - 1

        try:
            for t in range(self.T-M-2):
                if t+M > max_index or t+M+1 > max_index:
                    continue

                x = self.getData(t, M)
                w = strategy_func(x, **kwargs) if strategy_func != self.naiveStrategy else strategy_func()
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

            print(f'\n{self.name} - {strategy_name} - Window Size {M}:')
            print(f'Sharp ratio: insample = {SRI:.4f}, outpfsample = {SRO:.4f}')
            print(f'CEQ: outofsample = {CEQ:.6f}')
            print(f'Turnover: outofsample = {TO:.4f}')
            return {'SRI': SRI, 'SRO': SRO, 'CEQ': CEQ, 'TO': TO}

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

        portfolio_results = {
            'naive': {},
            'min_variance': {},
            'min_variance_constrained': {},
            'mean_variance_constrained': {},
            'bayes_stein_constrained': {},
            'g_min_variance_constrained': {}
        }

        for M in window_sizes:
            print(f'\nWindow size = {M}')
            
            # 评估1/N策略
            result = analyzer.evaluateStrategy(analyzer.naiveStrategy, "Naive (1/N)", M)
            if result:
                portfolio_results['naive'][M] = result
            
            # 评估最小方差策略（无约束）
            result = analyzer.evaluateStrategy(analyzer.minVarianceStrategy, "Minimum Variance", M)
            if result:
                portfolio_results['min_variance'][M] = result
            
            # 评估最小方差策略（有约束）
            result = analyzer.evaluateStrategy(analyzer.minVarianceStrategyConstrained, "Minimum Variance Constrained", M)
            if result:
                portfolio_results['min_variance_constrained'][M] = result
            
            # 评估均值-方差策略（有约束）
            result = analyzer.evaluateStrategy(analyzer.meanVarianceConstrainedStrategy, "Mean-Variance Constrained", M, risk_aversion=1.0)
            if result:
                portfolio_results['mean_variance_constrained'][M] = result
            
            # 评估贝叶斯-斯坦因策略（有约束）
            result = analyzer.evaluateStrategy(analyzer.bayesSteinConstrainedStrategy, "Bayes-Stein Constrained", M, risk_aversion=1.0)
            if result:
                portfolio_results['bayes_stein_constrained'][M] = result
            
            # 评估广义最小方差策略（有约束）
            result = analyzer.evaluateStrategy(analyzer.gMinVarianceConstrainedStrategy, "Generalized Min-Var Constrained", M, a=0.5)
            if result:
                portfolio_results['g_min_variance_constrained'][M] = result

            print("\nComparison of Out-of-Sample Sharpe Ratios:")
            for strategy, results in portfolio_results.items():
                if M in results:
                    print(f"{strategy}: {results[M]['SRO']:.4f}")
        
        results[portfolio['name']] = portfolio_results

    return results

if __name__ == '__main__':
    print("Starting Portfolio Analysis...")
    results = analyze_portfolios()
    print("\nAnalysis Complete!") 