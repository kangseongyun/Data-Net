import os
from scipy.stats import gamma, percentileofscore
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')

# 데이터 로드
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
data = pd.read_excel(file_path)

data = data[data['year_use'] == 2022]

data_n = pd.DataFrame()
data_n['매칭표제부PK'] = data['매칭표제부PK']
data_n['종별코드명'] = data['종별코드명']
data_n['연면적'] = data['연면적(㎡)']
data_n['층수'] = data['지상층수'] + data['지하층수']
data_n['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
data_n['의사수'] = data['총의사수']
data_n['병상수'] = data['총병상수']
data_n['USE_QTY_kWh'] = data['USE_QTY_kWh']

# Group titles
titles = [
    '전체병원',
    '종합병원',
    '병원 및 치과병원',
    '요양병원',
    '한방병원',
    '정신병원'
]

A2 = '종합병원'
A3 = '병원', '치과병원'
A4 = '요양병원'
A5 = '한방병원'
A6 = '정신병원'

data1 = data_n.copy()
X1 = data1[['연면적', "승강기수", "의사수", "병상수"]]
y1 = data1['USE_QTY_kWh']

data2 = data_n[data_n['종별코드명'].isin([A2])].copy()
X2 = data2[['연면적', "승강기수", "의사수"]]
y2 = data2['USE_QTY_kWh']

data3 = data_n[data_n['종별코드명'].isin(A3)].copy()
X3 = data3[['연면적', "승강기수", "의사수", "병상수"]]
y3 = data3['USE_QTY_kWh']

data4 = data_n[data_n['종별코드명'].isin([A4])].copy()
X4 = data4[['연면적', '층수', "의사수", "병상수"]]
y4 = data4['USE_QTY_kWh']

data5 = data_n[data_n['종별코드명'].isin([A5])].copy()
X5 = data5[['연면적', "의사수"]]
y5 = data5['USE_QTY_kWh']

data6 = data_n[data_n['종별코드명'].isin([A6])].copy()
X6 = data6[["의사수"]]
y6 = data6['USE_QTY_kWh']


def pred(A, B, C):
    X_with_const = sm.add_constant(A)
    model_unstandardized = sm.OLS(B, X_with_const).fit()
    print(model_unstandardized.summary())
    y_pred = model_unstandardized.predict(X_with_const)
    C.loc[:, 'pred'] = y_pred
    C.loc[:, 'eer'] = B / y_pred
    return C


eer_results = []
for X, y, data in zip([X1, X2, X3, X4, X5, X6], [y1, y2, y3, y4, y5, y6],
                      [data1, data2, data3, data4, data5, data6]):
    eer_results.append(pred(X, y, data))

efficiency_scores = [eer['eer'] for eer in eer_results]

# Create a grid of subplots (3 rows, 4 columns)
fig, axs = plt.subplots(3, 4, figsize=(24, 18))  # Adjust the figure size as needed

for i, scores in enumerate(efficiency_scores, start=0):
    sorted_efficiency_scores = np.sort(scores)
    cumulative_percent = np.array(
        [percentileofscore(sorted_efficiency_scores, x, 'rank') for x in sorted_efficiency_scores])

    # Cumulative Percent Plot
    ax_gamma = axs[i // 2, i % 2 * 2]  # First and third columns for cumulative percent
    ax_gamma.plot(sorted_efficiency_scores, cumulative_percent, label='Reference Data', marker='o', linestyle='None')
    ax_gamma.set_title(f'{titles[i]} - Cumulative Percent')  # Using titles for plot title
    ax_gamma.set_ylabel('Cumulative Percent')
    ax_gamma.grid(True)
    ax_gamma.set_xlim([0, 3])

    # Attempt to fit gamma distribution
    try:
        shape, loc, scale = gamma.fit(sorted_efficiency_scores, floc=0)

        # Gamma distribution curve
        x_vals = np.linspace(0, max(sorted_efficiency_scores), 1000)
        fitted_gamma = gamma.cdf(x_vals, shape, loc=loc, scale=scale) * 100

        # Plot fitted gamma curve
        ax_gamma.plot(x_vals, fitted_gamma, label='Fitted Curve', color='orange')
        ax_gamma.legend()
    except Exception as e:
        print(f"Gamma fitting failed for group {i + 1}: {e}")

    # Box Plot
    ax_box = axs[i // 2, i % 2 * 2 + 1]  # Second and fourth columns for box plot
    ax_box.boxplot(scores, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax_box.set_title(f'{titles[i]} - Box Plot')  # Using titles for plot title
    ax_box.grid(True)
    ax_box.set_xlim([0, 3])

plt.tight_layout()
plt.show()
