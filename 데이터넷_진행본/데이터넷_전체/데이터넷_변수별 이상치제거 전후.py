import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import skew, kurtosis

from scipy.stats import boxcox  # scipy.special이 아닌 scipy.stats에서 boxcox를 가져옵니다
from scipy.stats import pearsonr, shapiro, kstest, anderson, jarque_bera
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')


dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
q1 = pd.read_excel(file_path)
q1 = q1[q1['year_use'].isin([2022])]
def Factor(CZ):

    df_new = pd.DataFrame()
    df_new['종별코드명'] = CZ['종별코드명']
    df_new['연면적'] = CZ['연면적(㎡)']
    df_new['용적률산정연면적'] = CZ['용적률산정연면적(㎡)']
    df_new['대지면적'] = q1['대지면적(㎡)']
    df_new['층수'] = CZ['지상층수'] + CZ['지하층수']
    df_new['지하층수'] = CZ['지하층수']
    df_new['지상층수'] = CZ['지상층수']
    df_new['승강기수'] = CZ['비상용승강기수'] + CZ['승용승강기수']
    df_new['의사수'] = CZ['총의사수']
    df_new['병상수'] = CZ['총병상수']
    df_new['USE_QTY_kWh'] = CZ['USE_QTY_kWh']
    df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
    print('# of data1 : ', len(df_new))

    df_new = df_new[(df_new['대지면적'] > 0) & (df_new['의사수'] > 0) & (df_new['지상층수'] > 0)]
    print('# of data1 : ', len(df_new))
    return df_new








# 1사분위수(Q1)와 3사분위수(Q3) 계
def IQR_fitter(A):
    Q1 = A['USE_QTY_kWh'].quantile(0.25)
    Q3 = A['USE_QTY_kWh'].quantile(0.75)

    # IQR 계산
    IQR = Q3 - Q1

    # 이상치 경계값 계산
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치 제거
    q1 = A[(A['USE_QTY_kWh'] >= lower_bound) & (A['USE_QTY_kWh'] <= upper_bound)]



    return q1




q2=Factor(q1)




x=q2.drop(columns='USE_QTY_kWh')
y=q2['USE_QTY_kWh']

q3=IQR_fitter(q2)
filtered_x=q3.drop(columns='USE_QTY_kWh')
filtered_y=q3['USE_QTY_kWh']

# 이상치 제거 전후 산점도 플롯 함수
def plot_scatter_with_iqr_outlier_removal(X, y, filtered_x, filtered_y):
    for column in X.columns:
        original_x = X[column]
        filtered_x_col = filtered_x[column]  # 컬럼별로 필터링된 데이터를 참조
        original_y = y
        filtered_y_col = filtered_y

        # 1행 2열의 서브플롯
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 좌측: 이상치 제거 전
        axes[0].scatter(original_x, original_y, alpha=0.5)
        axes[0].set_title(f'Before Outlier Removal (IQR): {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('에너지 사용량')
        axes[0].grid(True)

        # 우측: 이상치 제거 후
        axes[1].scatter(filtered_x_col, filtered_y_col, alpha=0.5, color='orange')
        axes[1].set_title(f'After Outlier Removal (IQR): {column}')
        axes[1].set_xlabel(column)
        axes[1].set_ylabel('에너지 사용량')
        axes[1].grid(True)

        # 그래프 표시
        plt.tight_layout()
        plt.show()

# 함수 호출 (IQR 방법으로 이상치 제거 후 산점도 비교)
# plot_scatter_with_iqr_outlier_removal(x, y, filtered_x, filtered_y)




import seaborn as sns
import matplotlib.pyplot as plt

# 병원 종별에 따라 산점도를 그리는 함수 (체도가 높은 색상 사용)
def plot_scatter_with_iqr_outlier_removal_by_category(X, y, filtered_x, filtered_y, category, palette_type='bright'):
    unique_categories = category.nunique()  # 범주의 개수를 구함
    palette = sns.color_palette(palette_type, unique_categories)  # 범주 수에 맞게 팔레트 설정

    for column in X.columns:
        original_x = X[column]
        filtered_x_col = filtered_x[column]  # 컬럼별로 필터링된 데이터를 참조
        original_y = y
        filtered_y_col = filtered_y
        category_col = category  # 병원 종별

        # 1행 2열의 서브플롯
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 좌측: 이상치 제거 전
        sns.scatterplot(x=original_x, y=original_y, hue=category_col, palette=palette, ax=axes[0], alpha=0.7)
        axes[0].set_title(f'Before Outlier Removal (IQR): {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('에너지 사용량')
        axes[0].grid(True)

        # 우측: 이상치 제거 후
        sns.scatterplot(x=filtered_x_col, y=filtered_y_col, hue=category_col, palette=palette, ax=axes[1], alpha=0.7)
        axes[1].set_title(f'After Outlier Removal (IQR): {column}')
        axes[1].set_xlabel(column)
        axes[1].set_ylabel('에너지 사용량')
        axes[1].grid(True)

        # 범례 위치 및 크기 설정
        axes[0].legend(loc='upper right', title='종별코드명', fontsize='small', title_fontsize='medium')
        axes[1].legend(loc='upper right', title='종별코드명', fontsize='small', title_fontsize='medium')

        # 그래프 표시
        plt.tight_layout()
        plt.show()

# 병원 종별로 산점도 그리기 (체도가 높은 bright 팔레트 사용)
plot_scatter_with_iqr_outlier_removal_by_category(x, y, filtered_x, filtered_y, q2['종별코드명'], palette_type='bright')
