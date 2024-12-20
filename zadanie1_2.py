import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, kstest, mannwhitneyu, ttest_ind, shapiro, anderson, chi2, t, f
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

m = 3

file_path = 'MEN_SHOES.csv'
data = pd.read_csv(file_path)
data = data.dropna()
Y = np.array(data['RATING']).T
n = len(Y)
X1 = pd.to_numeric(data['How_Many_Sold'].replace({',': ''}, regex=True).values)  # Количество проданных
X2 = pd.to_numeric(data['Current_Price'].replace({'₹': '', ',': ''}, regex=True).values)  # Цена
X1 = X1.astype('float64')
X2 = X2.astype('float64')
X = np.column_stack((np.ones(n), X1, X2))
# Оценка параметров c
A = X.T @ X
c_hat = np.linalg.inv(A) @ X.T @ Y
# print(c_hat)
# Оценка остаточной дисперсии

S2 = (Y - X @ c_hat).T @ (Y - X @ c_hat)
# print(S2)
sigma2_hat = S2 / (n - m)
# print(sigma2_hat)
# Доверительный интервал для c_hat_i
df = n - m
alpha = 0.05

# t = t.ppf(1 - alpha/2, df)

A_inv = np.linalg.inv(A)
# for i in range(0, m):
#     down = c_hat[i] - np.sqrt((A_inv[i, i] * S2) / (n - m)) * t
#     up = c_hat[i] + np.sqrt((A_inv[i, i] * S2) / (n - m)) * t
#     print(down, up)

# t_nabl = c_hat[2] * np.sqrt((n - m) / (A_inv[2, 2] * S2))
# print(t_nabl)
# df = n - m  # Число степеней свободы
# alpha = 0.05
# t_critical = t.ppf(1 - alpha/2, df)
# print(t_critical)
c_hat = [3.34, 0, 0]
S2_H0 = (Y - X @ c_hat).T @ (Y - X @ c_hat)

f_nabl = (n - m) / m * (S2_H0 - S2) / S2

print(f_nabl)
alpha = 0.05
dfn = m     # числитель (число ограничений)
dfd = n - 2 # знаменатель (степени свободы остатка)

F_crit = f.ppf(1 - alpha, dfn, dfd)
print("F критическое:", F_crit)
print(F_crit)


# # Доверительный интервал для sigma
# down_sigma = S2 / chi2.ppf(alpha/2, df)
# up_sigma = S2 / chi2.ppf(1 - alpha/2, df)
# print(chi2.ppf(alpha/2, df),  chi2.ppf(1 - alpha/2, df))
# print(down_sigma, up_sigma)
#
#
