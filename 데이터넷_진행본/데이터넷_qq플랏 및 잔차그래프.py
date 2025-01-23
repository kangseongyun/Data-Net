import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import boxcox

# 데이터 로드 및 필터링
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
filtered_df = pd.read_excel(file_path)
print("# of PK3 : ", filtered_df['매칭표제부PK'].nunique())
print('# of data : ', filtered_df.shape[0])


# Group_3 = filtered_df[filtered_df['종별코드명'].isin(['정신병원','치과병원'])]
# Group_3_정신병원 = Group_3[Group_3['종별코드명'].isin(['정신병원'])]
# Group_3_치과병원 = Group_3[Group_3['종별코드명'].isin(['치과병원'])]
Group_3 = filtered_df[filtered_df['year_use'].isin([2022])]


# 새로운 데이터프레임 생성
df_new = pd.DataFrame()
df_new['매칭표제부PK'] = Group_3['매칭표제부PK']
df_new['승강기수'] = Group_3['비상용승강기수'] + filtered_df['승용승강기수']
df_new['의사수'] = Group_3['총의사수']
df_new['병상수'] = Group_3['총병상수']
df_new['용적률산정연면적'] = Group_3['용적률산정연면적(㎡)']
df_new['대지면적'] = Group_3['대지면적(㎡)']
df_new['연면적'] = Group_3['연면적(㎡)']
df_new['지하층수'] = Group_3['지하층수']
df_new['지상층수'] = Group_3['지상층수']
df_new['층수'] = Group_3['지상층수'] + Group_3['지하층수']
df_new['주용도비율'] = Group_3['주용도(의료시설) 비율(%)']
df_new['USE_QTY_kWh'] = Group_3['USE_QTY_kWh']
# df_new = df_new.sort_values(by='USE_QTY_kWh', ascending=True)
# df_new = df_new.sample(frac=1).reset_index(drop=True)

df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
plt.figure()

# boxplot
df_new[['USE_QTY_kWh']].boxplot()
plt.show()





plt.figure()
plt.hist(df_new['USE_QTY_kWh'], color = 'red', alpha = 0.4, bins=50)
plt.show()


#### IQR방법
# 1사분위수(Q1)와 3사분위수(Q3) 계산
Q1 = df_new['USE_QTY_kWh'].quantile(0.25)
Q3 = df_new['USE_QTY_kWh'].quantile(0.75)
# IQR 계산
IQR = Q3 - Q1
# 이상치 경계값 계산
lower_bound = Q1 - 1.5 * IQR
print(lower_bound)
upper_bound = Q3 + 1.5 * IQR
print(upper_bound)

# 이상치 제거
df_new = df_new[(df_new['USE_QTY_kWh'] >= lower_bound) & (df_new['USE_QTY_kWh'] <= upper_bound)]
plt.figure()
plt.hist(df_new['USE_QTY_kWh'], color = 'red', alpha = 0.4, bins=50)
plt.show()

def transform_hos(hos):
    hos = hos.copy()

    # # Box-Cox transformation of dependent variable
    hos['USE_QTY_kWh'], lambda_ = boxcox(hos['USE_QTY_kWh'])
    # print(lambda_)
    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)
    # df_new['USE_QTY_kWh'] = df_new['USE_QTY_kWh'].apply(lambda x: np.sqrt(x))
    return hos




df_new=transform_hos(df_new)
plt.figure()
plt.hist(df_new['USE_QTY_kWh'], color = 'red', alpha = 0.4, bins=50)
plt.show()

plt.figure()
# boxplot
df_new[['USE_QTY_kWh']].boxplot()
plt.show()

### ESD방법
#
# mean = np.mean(df_new['USE_QTY_kWh'])
# std = np.std(df_new['USE_QTY_kWh'])
#
# upper_bound = mean + 3*std
# lower_bound = mean - 3*std
# df_new = df_new[(df_new['USE_QTY_kWh'] >= lower_bound) & (df_new['USE_QTY_kWh'] <= upper_bound)]

# df_new['USE_QTY_kWh'], fitted_lambda = boxcox(df_new['USE_QTY_kWh'])  # 음수 또는 0 값이 없도록 미세한 값 추가
# print(fitted_lambda)
#
# plt.hist(df_new['USE_QTY_kWh'], bins=50, label='bins=50')
# plt.show()

independent_vars = ['승강기수','의사수',"연면적","대지면적",'층수']
X = df_new[independent_vars]
y = df_new['USE_QTY_kWh']

# 상수 추가
X = sm.add_constant(X)

# Fit regression model
model = sm.OLS(y, X).fit()

# Residuals and Fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Q-Q plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q plot of Residuals')
plt.show()

# Residuals vs Fitted values plot using sns.regplot
plt.figure(figsize=(8, 6))
sns.regplot(x=fitted_values, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.show()

test = sm.stats.omni_normtest(model.resid)
for xi in zip(['Chi^2', 'P-value'], test):
    print("%-12s: %6.3f" % xi)

# 정규성 검정 - Shapiro-Wilk Test
shapiro_test = stats.shapiro(model.resid)
print(f'Shapiro-Wilk Test: W={shapiro_test[0]}, p-value={shapiro_test[1]}')

# 정규성 검정 - Jarque-Bera Test
jarque_bera_test = stats.jarque_bera(model.resid)
print(f'Jarque-Bera Test: JB={jarque_bera_test[0]}, p-value={jarque_bera_test[1]}')

# 등분산성 검정 - Breusch-Pagan Test
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(f'Breusch-Pagan Test: {dict(zip(labels, bp_test))}')

# 회귀 결과 출력
print(model.summary())