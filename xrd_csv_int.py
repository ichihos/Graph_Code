import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import os

# CSVファイルを読み込む関数
def plot_xrd_data():
    # CSVファイルの読み込み
    data_files = [os.path.join('../data', file) for file in os.listdir('../data') if file.endswith('.csv')]
    int_files = [os.path.join('../data', file) for file in os.listdir('../data') if file.endswith('.int')]
    fig, axs = plt.subplots(2, 1, figsize=(25, 10), sharex=True)
    for file in data_files:
        data = pd.read_csv(file, skiprows=26)
        xraw = data.iloc[:, 0]
        yraw = data.iloc[:, 1]
        y = yraw[(xraw >= 20) & (xraw <= 100)]
        x = xraw[(xraw >= 20) & (xraw <= 100)]
        coefficients = Polynomial.fit(x, y, 5).convert().coef
        background = np.polyval(coefficients[::-1], x)
        corrected_intensities = y - background
        corrected_intensities[corrected_intensities < 0] = 0
        smoothed_intensities = corrected_intensities.rolling(window=8, center=True).mean()
        axs[0].plot(x,smoothed_intensities, label=f"{os.path.basename(file)[:-4]}")
        axs[0].set_ylabel("Intensity (a.u.)")
        axs[0].grid(True)
        axs[0].legend(loc="upper right")
    for file in int_files:
        data = pd.read_csv(file, delim_whitespace=True,skiprows=26)
        yraw = data.iloc[:, 1]
        xraw = data.iloc[:, 0]
        y = yraw[(xraw >= 20) & (xraw <= 100)]
        x = xraw[(xraw >= 20) & (xraw <= 100)]
        coefficients = Polynomial.fit(x, y, 5).convert().coef
        background = np.polyval(coefficients[::-1], x)
        corrected_intensities = y - background
        corrected_intensities[corrected_intensities < 0] = 0
        smoothed_intensities = corrected_intensities.rolling(window=8, center=True).mean()
        axs[1].plot(x,y, label=f"{os.path.basename(file)[:-4]}_calc")
        axs[1].set_xlabel("2θ (Degrees)")
        axs[1].set_ylabel("Intensity (a.u.)")
        axs[1].grid(True)
        axs[1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()
    fig.savefig('../output/xrd_csvint.png')

# グラフを描画
plot_xrd_data()
