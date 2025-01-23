import os
from scipy.stats import pearsonr

from scipy.stats import boxcox  # scipy.special이 아닌 scipy.stats에서 boxcox를 가져옵니다
import numpy as np

from scipy.stats import gamma
from scipy.stats import percentileofscore
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')
# 데이터 로드
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
data = pd.read_excel(file_path)
# data = data[data['주용도(의료시설) 비율(%)'] >= 90]

print('Before : 필터')
print("# of PK1 : ", data['mgm_bld_pk'].nunique())  # of PK1 :    3562
print('# of data1 : ', len(data))  # of data1 :  3967
print(' ')


data = data[data['year_use'] == 2022]
data['종별코드명'] = data['종별코드명'].str.strip()

A= 'Total'
A= '종합병원'
A= '병원'
A= '요양병원'
A= '한방병원'
A= '정신병원'


if A in 'Total':
    data = data[(data['종별코드명'] == '치과병원')]
else:
    data = data[data['종별코드명'].isin([A])]



print('# of data2 : ', len(data))  # of data1 :  3967
print(' ')


# 필요한 변수 생성
data_n = pd.DataFrame()

data_n['연면적'] = data['연면적(㎡)']
data_n['층수'] = data['지상층수']+data['지하층수']
data_n['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
data_n['의사수'] = data['총의사수']
data_n['병상수'] = data['총병상수']



data_n['USE_QTY_kWh'] = data['USE_QTY_kWh']

print('# of data1 : ', len(data_n))  # of data1 :  3967
print(' ')



# 설명 변수와 목표 변수 설정

if A in 'Total':
    X = data_n[['연면적', "승강기수", "의사수", "병상수"]]
if A in '종합병원':
    X = data_n[['연면적', "승강기수", "의사수"]]
if A in '병원':
    X = data_n[['연면적', "승강기수", "의사수"]]
if A in '요양병원':
    X = data_n[['연면적', '층수', "의사수", "병상수"]]
if A in '한방병원':
    X = data_n[['연면적', "의사수"]]
if A in '정신병원':
    X = data_n[["의사수"]]

y = data_n['USE_QTY_kWh']



# 비표준화된 데이터로 회귀 분석
X_with_const = sm.add_constant(X)
model_unstandardized = sm.OLS(y, X_with_const).fit()



y_pred = model_unstandardized.predict(X_with_const)

efficiency_scores = 100 * (y_pred / y)





# 효율성 점수를 정렬
sorted_efficiency_scores = np.sort(efficiency_scores)

# 각 효율성 점수에 대한 누적 백분율 계산
cumulative_percent = np.array([percentileofscore(sorted_efficiency_scores, x, 'rank') for x in sorted_efficiency_scores])

# Gamma 분포 피팅
shape, loc, scale = gamma.fit(sorted_efficiency_scores, floc=0)  # loc 고정

# Gamma 분포에 따른 누적 분포 곡선 생성
x_vals = np.linspace(0, max(sorted_efficiency_scores), 1000)
fitted_gamma = gamma.cdf(x_vals, shape, loc=loc, scale=scale) * 100  # 퍼센트로 변환

# 누적 분포와 피팅된 Gamma 곡선 그리기
plt.figure(figsize=(8, 6))
plt.plot(sorted_efficiency_scores, cumulative_percent, label='Reference Data', marker='o', linestyle='None')
plt.plot(x_vals, fitted_gamma, label='Fitted Curve', color='orange')
plt.xlabel('Efficiency Ratio (Predicted USE_QTY_kWh / Actual USE_QTY_kWh)')
plt.ylabel('Cumulative Percent')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()






# 표준화된 데이터로 회귀 분석
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

X_scaled_with_const = sm.add_constant(X_scaled)
model_standardized = sm.OLS(y_scaled, X_scaled_with_const).fit()

# 비표준화 회귀 결과 출력
print("비표준화 회귀 계수 및 요약 통계:")
print(model_unstandardized.summary())

# 표준화된 회귀 계수 출력
standardized_coefficients = model_standardized.params
print("\n표준화 회귀계수:")
for var, coef in zip(['const', '용적률산정연면적', '층수', '의사수'], standardized_coefficients):
    print(f'{var}: {coef}')