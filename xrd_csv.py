
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import os

# CSVファイルを読み込む関数
def plot_xrd_data():
    # CSVファイルの読み込み
    data_files = [os.path.join('../data', file) for file in os.listdir('../data') if file.endswith('.csv')]
    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
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
        plt.plot(x,smoothed_intensities, label=f"{os.path.basename(file)[:-4]}")
        plt.grid(True)
        ax.set_xlabel("2θ (Degrees)")
        ax.set_ylabel("Intensity (a.u.)")
        plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()
    fig.savefig('../output/xrd_csv.png')

# グラフを描画
plot_xrd_data()
