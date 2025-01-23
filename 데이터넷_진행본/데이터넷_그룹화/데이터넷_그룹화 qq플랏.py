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

data = filtered_df[filtered_df['year_use'] == 2022]

Group_1 = data[data['종별코드명'].isin(['종합병원'])]

Group_2 = data[data['종별코드명'].isin(['병원','요양병원'])]

Group_3 = data[data['종별코드명'].isin(["한방병원",'정신병원','치과병원'])]

data=Group_1
# method='기본'
method='이상치'

# 새로운 데이터프레임 생성
df_new = pd.DataFrame()
df_new['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
df_new['의사수'] = data['총의사수']
df_new['병상수'] = data['총병상수']
df_new['용적률산정연면적'] = data['용적률산정연면적(㎡)']
df_new['대지면적'] = data['대지면적(㎡)']
df_new['연면적'] = data['연면적(㎡)']
df_new['지하층수'] = data['지하층수']
df_new['지상층수'] = data['지상층수']
df_new['층수'] = data['지상층수'] + data['지하층수']
df_new['USE_QTY_kWh'] = data['USE_QTY_kWh']
# df_new = df_new.sample(frac=1).reset_index(drop=True)

df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])





#### IQR방법

def transform_hos(hos,method1):
    hos = hos.copy()

    if method1 in '이상치':
        # # 1사분위수(Q1)와 3사분위수(Q3) 계산
        Q1 = hos['USE_QTY_kWh'].quantile(0.25)
        Q3 = hos['USE_QTY_kWh'].quantile(0.75)
        # IQR 계산
        IQR = Q3 - Q1
        # 이상치 경계값 계산
        lower_bound = Q1 - 1.5 * IQR
        print(lower_bound)
        upper_bound = Q3 + 1.5 * IQR
        print(upper_bound)

        hos = hos[(hos['USE_QTY_kWh'] >= lower_bound) & (hos['USE_QTY_kWh'] <= upper_bound)]


    # # # # Box-Cox transformation of dependent variable
    # hos['USE_QTY_kWh'], lambda_ = boxcox(hos['USE_QTY_kWh'])
    # print(lambda_)
    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)
    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.sqrt(x))
    return hos




df_new=transform_hos(df_new,method)
if method == '기본':
    if data.equals(Group_1):
        independent_vars = ['연면적','대지면적',"승강기수","의사수"]
    elif data.equals(Group_2):
        independent_vars = ['연면적','승강기수',"의사수"]
    elif data.equals(Group_3):
        independent_vars = ['연면적',"의사수"]

elif method == '이상치':
    if data.equals(Group_1):
        independent_vars = ['용적률산정연면적', '층수', "의사수"]
    elif data.equals(Group_2):
        independent_vars = ['연면적','지상층수',"의사수","병상수"]
    elif data.equals(Group_3):
        independent_vars = ['연면적','대지면적']




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