import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, kstest, mannwhitneyu, ttest_ind, shapiro, anderson, chi2, t, f
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

file_path = 'iris.csv'
data = pd.read_csv(file_path)
data['Summary_width'] = data['Sepal.Length'] * data['Sepal.Width'] + data['Petal.Length'] * data['Petal.Width']
unique_species = data['Species'].unique()

filtered_data = data[data['Species'] == 'setosa']
y1 = np.mean(filtered_data['Summary_width'])
filtered_data = data[data['Species'] == 'versicolor']
y2 = np.mean(filtered_data['Summary_width'])
filtered_data = data[data['Species'] == 'virginica']
y3 = np.mean(filtered_data['Summary_width'])
y_sr = np.array([y1, y2, y3])
Sb = 0
for j in range (0, 3):
    filtered_data = data[data['Species'] == unique_species[j]]
    Sb += len(filtered_data) * (y_sr[j] - np.mean(y_sr)) ** 2
MSb = Sb / (len(unique_species) - 1)
# print(MSb)
Sw = 0
for j in range(0, 3):
    filtered_data = data[data['Species'] == unique_species[j]]
    filtered_data = filtered_data['Summary_width'].values
    for i in range(50):
        Sw += (filtered_data[i] - y_sr[j])**2
MSw = Sw / (len(data) - len(unique_species))
alpha = 0.05
dfn = len(unique_species)     # числитель (число ограничений)
dfd = len(data) - len(unique_species) # знаменатель (степени свободы остатка)

F_crit = f.ppf(1 - alpha, dfn, dfd)
print("F критическое:", F_crit)