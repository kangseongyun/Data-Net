import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations

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

#
# plt.figure()
#
# # boxplot
# q1[['USE_QTY_kWh']].boxplot()
# plt.ylabel('USE_QTY_kWh')
# plt.tight_layout()
# plt.show()


# 1사분위수(Q1)와 3사분위수(Q3) 계
def IQR_fitter(q1):
    Q1 = q1['USE_QTY_kWh'].quantile(0.25)
    Q3 = q1['USE_QTY_kWh'].quantile(0.75)

    # IQR 계산
    IQR = Q3 - Q1

    # 이상치 경계값 계산
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(lower_bound, upper_bound)

    # 이상치 제거
    q1 = q1[(q1['USE_QTY_kWh'] >= lower_bound) & (q1['USE_QTY_kWh'] <= upper_bound)]

    return q1
# q1 = IQR_fitter(q1)

# q1['USE_QTY_kWh'] = q1['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)

# plt.figure()
#
# # boxplot
# q1[['USE_QTY_kWh']].boxplot()
# plt.ylabel('USE_QTY_kWh')
# plt.tight_layout()
# plt.show()


hos_1=q1[q1['year_use']==2022]
# hos_1 = q1[q1['종별코드명']=='병원'].copy()
# hos_2 = q1[q1['종별코드명']=='치과병원'].copy()
# hos_3 = q1[q1['종별코드명']=='한방병원'].copy()
# hos_4 = q1[q1['종별코드명']=='요양병원'].copy()
# hos_5 = q1[q1['종별코드명']=='정신병원'].copy()
# hos_6 = q1[q1['종별코드명']=='종합병원'].copy()



def transform_hos(hos):
    hos=hos.copy()
    # Box-Cox transformation of dependent variable
    hos['USE_QTY_kWh'], lambda_ = boxcox(hos['USE_QTY_kWh'])
    print(lambda_)
    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.log(x))
    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.sqrt(x))
    return hos



hos_1=IQR_fitter(hos_1)
# hos_2=IQR_fitter(hos_2)
# hos_3=IQR_fitter(hos_3)
# hos_4=IQR_fitter(hos_4)
# hos_5=IQR_fitter(hos_5)
# hos_6=IQR_fitter(hos_6)


hos_1=transform_hos(hos_1)
# hos_2=transform_hos(hos_2)
# hos_3=transform_hos(hos_3)
# hos_4=transform_hos(hos_4)
# hos_5=transform_hos(hos_5)
# hos_6=transform_hos(hos_6)


# hos_7 = q1[q1['종별코드명'].isin(['병원','치과병원', '한방병원', '요양병원', '정신병원', '종합병원'])]
hos_7 = q1[q1['종별코드명'].isin(['병원','치과병원', '한방병원', '요양병원', '정신병원'])]
sns.histplot(hos_1['USE_QTY_kWh'], kde= True, color = 'red', alpha = 0.4, bins=50, label='Total')

# plt.hist(hos_1['USE_QTY_kWh'], color = 'red', alpha = 0.4, bins=50, label='병원')
# plt.hist(hos_2['USE_QTY_kWh'], color = 'green', alpha = 0.4, bins=50, label='치과병원')
# plt.hist(hos_3['USE_QTY_kWh'], color = 'blue', alpha = 0.4, bins=50, label='한방병원')
# plt.hist(hos_4['USE_QTY_kWh'], color = 'yellow', alpha = 0.4, bins=50, label='요양병원')
# plt.hist(hos_5['USE_QTY_kWh'], color = 'cyan', alpha = 0.4, bins=50, label='정신병원')
# plt.hist(hos_6['USE_QTY_kWh'], color = 'gray', alpha = 0.4, bins=50, label='종합병원')
# plt.hist(hos_7['USE_QTY_kWh'], color = 'black', alpha = 0.4, bins=50, label='종합병원 제외')

plt.legend()
plt.xlabel('USE_QTY_kWh')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()





# #
# plt.xlim(0, 1750)
# plt.ylim(0, 140)

# log
# plt.xlim(6, 19)
# plt.ylim(0, 300)


# IQR
# plt.xlim(0, 3000000)
# plt.ylim(0, 130)
#
# # IQR_log
# plt.xlim(6, 16)
# plt.ylim(0, 300)


## log_IQR
# plt.xlim(11.5, 16)
# plt.ylim(0, 130)


# # Total
# plt.xlim(6, 19)
# plt.ylim(0, 600)




#
#
# # 종속 변수 로그 변환
# def scatter(A):
#     df_new = pd.DataFrame()
#     df_new['승강기수'] = A['비상용승강기수'] + A['승용승강기수']  # all
#     df_new['의사수'] = A['총의사수']  # all
#     df_new['병상수'] = A['총병상수']  # all
#     df_new['용적률산정연면적'] = A['용적률산정연면적(㎡)']
#     df_new['대지면적'] = A['대지면적(㎡)']  # all
#     df_new['연면적'] = A['연면적(㎡)']  # all
#     df_new['지하층수'] = A['지하층수']  # all
#     df_new['지상층수'] = A['지상층수']  # all
#     df_new['층수'] = df_new['지상층수'] + df_new['지하층수']
#     # df_new['주용도비율'] = filtered_df['주용도(의료시설) 비율(%)']
#     df_new['USE_QTY_kWh'] = A['USE_QTY_kWh']
#
#
#
#     df_new['승강기수'] = df_new['승강기수'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['의사수'] = df_new['의사수'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['병상수'] = df_new['병상수'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['용적률산정연면적'] = df_new['용적률산정연면적'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['대지면적'] = df_new['대지면적'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['연면적'] = df_new['연면적'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['지하층수'] = df_new['지하층수'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['지상층수'] = df_new['지상층수'].apply(lambda x: np.log(x) if x > 0 else 0)
#     df_new['층수'] = df_new['층수'].apply(lambda x: np.log(x) if x > 0 else 0)
#     # df_new['USE_QTY_kWh'] = df_new['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)
#
#
#     # df_new['승강기수'] = df_new['승강기수'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['의사수'] = df_new['의사수'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['병상수'] = df_new['병상수'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['용적률산정연면적'] = df_new['용적률산정연면적'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['대지면적'] = df_new['대지면적'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['연면적'] = df_new['연면적'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['지하층수'] = df_new['지하층수'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['지상층수'] = df_new['지상층수'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['층수'] = df_new['층수'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#     # df_new['USE_QTY_kWh'] = df_new['USE_QTY_kWh'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#
#
#
#
#     independent_vars = ['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수']
#
#     results = []
#
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 25))
#     axes = axes.flatten()
#
#     for idx, var in enumerate(independent_vars):
#         # sns.regplot(x=df_new[var], y=df_new['log_USE_QTY_kWh'], ax=axes[idx], ci=None)
#         sns.regplot(x=df_new[var], y=df_new['USE_QTY_kWh'], ax=axes[idx], ci=None)
#         axes[idx].set_title(f'{var} vs USE_QTY_kWh', fontsize=20)  # 제목 글자 크기 설정
#         axes[idx].set_xlabel(var, fontsize=15)  # x축 레이블 글자 크기 설정
#         axes[idx].set_ylabel('USE_QTY_kWh', fontsize=15)  # y축 레이블 글자 크기 설정
#
#         # 회귀 분석
#         X = df_new[[var]]
#         y = df_new['USE_QTY_kWh']
#         model = sm.OLS(y, sm.add_constant(X)).fit()
#
#         # 회귀 계수 및 R^2 값 저장
#         results.append({
#             '변수': var,
#             '회귀 계수': model.params[1],
#             'R^2': model.rsquared
#         })
#
#         # 회귀선과 회귀식, R^2 값 표시
#         intercept = model.params[0]
#         slope = model.params[1]
#         r_squared = model.rsquared
#         eq = f'y = {intercept:.2f} + {slope:.2f}x'
#         r2_text = f'R² = {r_squared:.2f}'
#
#         axes[idx].text(0.05, 0.95, eq, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')
#         axes[idx].text(0.05, 0.85, r2_text, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')
#
#     plt.tight_layout()
#     plt.show()
# scatter(hos_1)