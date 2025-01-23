import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from matplotlib import pyplot as plt
from scipy.stats import shapiro, kstest, anderson, jarque_bera, normaltest, boxcox
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')


# 데이터 경로 및 파일명 설정
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)

# 엑셀 파일 읽기
q1 = pd.read_excel(file_path)

# 연도 설정

# 데이터 필터링
Group_1 = q1[q1['종별코드명'].isin(['종합병원'])]

Group_2 = q1[q1['종별코드명'].isin(['병원','요양병원'])]

Group_3 = q1[q1['종별코드명'].isin(["한방병원",'정신병원','치과병원'])]


dir_path_porder1 = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본\그룹화"

year = 2022


dir_path_porder3 = "Group_1"
Analysis_target=Group_1

# dir_path_porder3 = "Group_2"
# Analysis_target=Group_2
#
# dir_path_porder3 = "Group_3"
# Analysis_target=Group_3


dir_path_porder_method = "기본"
# dir_path_porder_method = "sqrt"
# dir_path_porder_method = "log"
# dir_path_porder_method = "boxcox"
# dir_path_porder_method = "IQR"
# dir_path_porder_method = "IQR_sqrt"
# dir_path_porder_method = "IQR_log"
# dir_path_porder_method = "IQR_boxcox"



def transform_hos(hos, dir_path_porder_method):
    hos = hos.copy()
    #
    if dir_path_porder_method in 'IQR':
        Q1 = hos['USE_QTY_kWh'].quantile(0.25)
        Q3 = hos['USE_QTY_kWh'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        hos = hos[(hos['USE_QTY_kWh'] >= lower_bound) & (hos['USE_QTY_kWh'] <= upper_bound)]
    else:
        hos=hos

    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.sqrt(x) if x > 0 else 0)

    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)

    # Box-Cox transformation of dependent variable
    # hos['USE_QTY_kWh'], lambda_ = boxcox(hos['USE_QTY_kWh'])
    # print(lambda_)
    return hos


Analysis_target = Analysis_target[Analysis_target['year_use'].isin([year])]

df_new = pd.DataFrame()
df_new['연면적'] = Analysis_target['연면적(㎡)']
df_new['용적률산정연면적'] = Analysis_target['용적률산정연면적(㎡)']
df_new['대지면적'] = Analysis_target['대지면적(㎡)']
df_new['층수'] = Analysis_target['지상층수'] + Analysis_target['지하층수']
df_new['지하층수'] = Analysis_target['지하층수']
df_new['지상층수'] = Analysis_target['지상층수']
df_new['승강기수'] = Analysis_target['비상용승강기수'] + Analysis_target['승용승강기수']
df_new['의사수'] = Analysis_target['총의사수']
df_new['병상수'] = Analysis_target['총병상수']



df_new['USE_QTY_kWh'] = Analysis_target['USE_QTY_kWh']

df_new=transform_hos(df_new,dir_path_porder_method)
df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])


independent_vars = ['연면적', '용적률산정연면적', '대지면적', '층수','지하층수', '지상층수', '승강기수', '의사수', '병상수']

results = []

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 25))
axes = axes.flatten()

for idx, var in enumerate(independent_vars):
    # sns.regplot(x=df_new[var], y=df_new['log_USE_QTY_kWh'], ax=axes[idx], ci=None)
    sns.regplot(x=df_new[var], y=df_new['USE_QTY_kWh'], ax=axes[idx], ci=None)
    axes[idx].set_title(f'{var} vs USE_QTY_kWh', fontsize=20)  # 제목 글자 크기 설정
    axes[idx].set_xlabel(var, fontsize=15)  # x축 레이블 글자 크기 설정
    axes[idx].set_ylabel('USE_QTY_kWh', fontsize=15)  # y축 레이블 글자 크기 설정

    # 회귀 분석
    X = df_new[[var]]
    y = df_new['USE_QTY_kWh']
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # 회귀 계수 및 R^2 값 저장
    results.append({
        '변수': var,
        '회귀 계수': model.params[1],
        'R^2': model.rsquared
    })

    # 회귀선과 회귀식, R^2 값 표시
    intercept = model.params[0]
    slope = model.params[1]
    r_squared = model.rsquared
    eq = f'y = {intercept:.2f} + {slope:.2f}x'
    r2_text = f'R² = {r_squared:.2f}'

    axes[idx].text(0.05, 0.95, eq, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')
    axes[idx].text(0.05, 0.85, r2_text, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')

plt.tight_layout()
plt.show()

# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame(results)
#
# # 엑셀 파일로 저장
# output_filename = "회귀분석결과_log_USE_QTY_kWh.xlsx"
# output_filepath = os.path.join(dir_path, output_filename)
# results_df.T.to_excel(output_filepath, index=True)