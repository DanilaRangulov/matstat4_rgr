import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, kstest, mannwhitneyu, ttest_ind, shapiro, anderson, chi2, t
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')

m = 2

file_path = 'MEN_SHOES.csv'
data = pd.read_csv(file_path)
data = data.dropna()
Y = np.array(data['RATING']).T
n = len(Y)
X1 = pd.to_numeric(data['How_Many_Sold'].replace({',': ''}, regex=True).values)  # Количество проданных
X2 = pd.to_numeric(data['Current_Price'].replace({'₹': '', ',': ''}, regex=True).values)  # Цена
X1 = X1.astype('float64')
X2 = X2.astype('float64')
X = np.vstack((X1, X2)).T


model = LinearRegression(fit_intercept=True)
model.fit(X, Y)

# Коэффициенты:
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Предсказания:
Y_pred = model.predict(X)

# RSS:
RSS = ((Y - Y_pred)**2).sum()

# TSS:
TSS = ((Y - Y.mean())**2).sum()

print('RSS', RSS)
print('TSS', TSS)

R2 = 1 - RSS / TSS
print(R2)

# Создаем сетку для X1 и X2
x1_grid = np.linspace(X1.min(), X1.max(), 1000)
x2_grid = np.linspace(X2.min(), X2.max(), 1000)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

# Вычисляем предсказания на сетке
Y_pred_surface = model.predict(np.array([X1_grid.ravel(), X2_grid.ravel()]).T)
Y_pred_surface = Y_pred_surface.reshape(X1_grid.shape)

# Создаем 3D фигуру
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Исходные данные в виде точек
ax.scatter(X1, X2, Y, c='r', marker='o', alpha=0.6, label='Исходные данные')

# Поверхность регрессии
ax.plot_surface(X1_grid, X2_grid, Y_pred_surface, alpha=0.5, cmap='viridis')

ax.set_xlabel('X1 (Количество проданных)')
ax.set_ylabel('X2 (Текущая цена)')
ax.set_zlabel('Y (Рейтинг)')

plt.title('3D визуализация линейной регрессии')
plt.legend()
plt.show()


