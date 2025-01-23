import os
import itertools
from openpyxl import Workbook
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, kstest, anderson, jarque_bera, normaltest, boxcox, skew, kurtosis

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')
# 데이터 로드
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
data = pd.read_excel(file_path)
data = data[data['주용도(의료시설) 비율(%)'] >= 90]

data = data[data['year_use'] == 2022]

data = data[(data['종별코드명'] != '치과병원')]
# data = data[data['종별코드명'].isin(['종합병원'])]
# data = data[data['종별코드명'].isin(['병원'])]
# data = data[data['종별코드명'] == '병원']
# data = data[data['종별코드명'].isin(['요양병원'])]
# data = data[data['종별코드명'].isin(['한방병원'])]
# data = data[data['종별코드명'].isin(['정신병원'])]
# data = data[data['종별코드명'].isin(['치과병원'])]


output_path = r"C:\Users\user\Desktop\다중회귀결과\Total.xlsx"
# output_path = r"C:\Users\user\Desktop\다중회귀결과\종합병원.xlsx"
# output_path = r"C:\Users\user\Desktop\다중회귀결과\병원.xlsx"
# output_path = r"C:\Users\user\Desktop\다중회귀결과\요양병원.xlsx"
# output_path = r"C:\Users\user\Desktop\다중회귀결과\한방병원.xlsx"
# output_path = r"C:\Users\user\Desktop\다중회귀결과\정신병원.xlsx"


method='기본'
# method='IQR'


# 필요한 변수 생성
data_n = pd.DataFrame()
data_n['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
data_n['의사수'] = data['총의사수']
data_n['병상수'] = data['총병상수']
data_n['연면적'] = data['연면적(㎡)']
data_n['용적률산정연면적'] = data['용적률산정연면적(㎡)']
data_n['대지면적'] = data['대지면적(㎡)']
data_n['지하층수'] = data['지하층수']
data_n['지상층수'] = data['지상층수']
data_n['층수'] = data['지상층수']+data['지하층수']
data_n['USE_QTY_kWh'] = data['USE_QTY_kWh']

data_n = data_n.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
print('# of data1 : ', len(data_n))



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


    return hos

data_n=transform_hos(data_n, method)

print('# of data1 : ', len(data_n))

def centering_data(a):
    c_data = pd.DataFrame()
    c_data['승강기수'] = a['승강기수']-a['승강기수'].mean()
    print(data_n['승강기수'].mean())
    c_data['의사수'] = a['의사수']-a['의사수'].mean()
    c_data['병상수'] = a['병상수']-a['병상수'].mean()
    c_data['연면적'] = a['연면적']-a['연면적'].mean()
    c_data['용적률산정연면적'] = a['용적률산정연면적']-a['용적률산정연면적'].mean()
    c_data['대지면적'] = a['대지면적']-a['대지면적'].mean()
    c_data['지하층수'] = a['지하층수']-a['지하층수'].mean()
    c_data['지상층수'] = a['지상층수']-a['지상층수'].mean()
    c_data['층수'] = a['층수']-a['층수'].mean()
    c_data['USE_QTY_kWh'] = a['USE_QTY_kWh']
    return c_data


c_data=centering_data(data_n)

X = c_data[['연면적','층수','병상수','의사수','승강기수']]


def run_regression_and_save_results(X, y, output_path):
    results = []

    # X의 모든 가능한 조합을 순회
    for L in range(1, len(X.columns) + 1):
        for subset in itertools.combinations(X.columns, L):
            X_subset = X[list(subset)]
            X_subset = sm.add_constant(X_subset)  # 상수항 추가

            # 회귀 모델 적합
            model = sm.OLS(y, X_subset).fit()
            print(model.summary())
            residuals = model.resid

            # VIF 계산 (상수항 제외)
            vif = pd.Series([variance_inflation_factor(X_subset.values, i) for i in range(X_subset.shape[1])],
                            index=X_subset.columns)

            # 표준화 회귀 계수 계산 (상수항 제외)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset.iloc[:, 1:])  # 상수항을 제외한 나머지 변수만 표준화
            y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()  # y를 1차원으로 변환

            X_scaled_with_const = sm.add_constant(X_scaled)  # 상수항을 추가한 표준화 데이터
            model_standardized = sm.OLS(y_scaled, X_scaled_with_const).fit()

            std_coef = model_standardized.params

            # 결과 저장
            result = {
                'Variables': ' + '.join(subset),
                'AIC': model.aic,
                'BIC': model.bic,
                'R-squared': model.rsquared,
                'Adjusted R-squared': model.rsquared_adj,
                'Skewness': skew(residuals),
                'Kurtosis': kurtosis(residuals),
                'Durbin-Watson': sm.stats.durbin_watson(residuals),
                'Prob (F-statistic)': model.f_pvalue,
                'F-statistic': model.fvalue,
                'No. Observations': model.nobs,
            }

            # 상수항 관련 결과 추가
            result['Coef_const'] = model.params['const']  # 상수항의 회귀 계수
            result['p-value_const'] = model.pvalues['const']
            result['Std err_const'] = model.bse['const']
            result['T-value_const'] = model.tvalues['const']  # 상수항의 t-value

            # 각 변수별 결과 추가 (상수항 제외)
            for i, var in enumerate(subset):
                result[f'Coef_{var}'] = model.params[var]
                result[f'Std_Coef_{var}'] = std_coef[i + 1]  # 표준화 회귀 계수 (상수항 제외)
                result[f'T-value_{var}'] = model.tvalues[var]  # 각 변수의 t-value
                result[f'p-value_{var}'] = model.pvalues[var]
                result[f'VIF_{var}'] = vif[var]
                result[f'Std err_{var}'] = model.bse[var]

            results.append(result)

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)

    # 결과를 엑셀로 저장
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")




# 데이터 전처리 및 설명 변수, 목표 변수 설정
# X = data_n[['용적률산정연면적', '층수', '의사수']]
# X = data_n[['연면적', '대지면적']]

y = c_data['USE_QTY_kWh']



# 모든 조합에 대한 회귀 분석 수행 및 결과 저장
run_regression_and_save_results(X, y, output_path)