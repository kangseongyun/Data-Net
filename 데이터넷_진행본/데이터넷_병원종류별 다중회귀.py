import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import shapiro, kstest, anderson, jarque_bera
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan

# 데이터 경로 및 파일명 설정d
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)

# 엑셀 파일 읽기
q1 = pd.read_excel(file_path)
q1 = q1[q1['주용도(의료시설) 비율(%)'] >= 90]

# 연도 설정
year = 2022

# 데이터 필터링
hos_1 = q1[q1['종별코드명'].isin(['병원', '치과병원', '한방병원', '요양병원', '정신병원'])]
A="종합병원"
hos_1 = q1[q1['종별코드명'].isin([A])]
# hos_2 = hos_2.sample(frac=1).reset_index(drop=True)

hos_1 = hos_1[hos_1['year_use'] == year]

# 회귀 분석을 위한 데이터 준비
def multiple_regression(A):
    df_new = pd.DataFrame()
    df_new['승강기수'] = A['비상용승강기수'] + A['승용승강기수']
    df_new['의사수'] = A['총의사수']
    df_new['병상수'] = A['총병상수']
    # df_new['용적률산정연면적'] = A['용적률산정연면적(㎡)']
    # df_new['대지면적'] = A['대지면적(㎡)']
    df_new['연면적'] = A['연면적(㎡)']
    # df_new['지하층수'] = A['지하층수']
    # df_new['지상층수'] = A['지상층수']
    df_new['층수'] = A['지상층수'] + A['지하층수']
    df_new['USE_QTY_kWh'] = A['USE_QTY_kWh']

    # Q1 = df_new['USE_QTY_kWh'].quantile(0.25)
    # Q3 = df_new['USE_QTY_kWh'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # df_new = df_new[(df_new['USE_QTY_kWh'] >= lower_bound) & (df_new['USE_QTY_kWh'] <= upper_bound)]

    # df_new['USE_QTY_kWh'] = df_new['USE_QTY_kWh'].apply(lambda x: np.sqrt(x) if x > 0 else 0)

    df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '연면적',  '층수'])
    return df_new

# R2, VIF 계산 함수
def calculate_individual_r2_vif(data, predictors):
    r2_values = {}
    vif_values = {}
    tolerance_values = {}
    for predictor in predictors:
        other_predictors = [p for p in predictors if p != predictor]
        if len(other_predictors) > 0:
            formula = "{} ~ {}".format(predictor, ' + '.join(other_predictors))
            model = sm.OLS.from_formula(formula, data).fit()
            r2 = model.rsquared
            r2_values[predictor] = round(r2, 4)
            tolerance = 1 - r2
            tolerance_values[predictor] = round(tolerance, 4)
            vif_values[predictor] = round(1 / tolerance, 4) if tolerance > 0 else float('inf')
        else:
            r2_values[predictor] = float('nan')
            vif_values[predictor] = float('nan')
            tolerance_values[predictor] = float('nan')
    return r2_values, vif_values, tolerance_values

# 최적 모델 탐색 함수
def get_best_model(data, response):
    predictors = data.columns.tolist()
    predictors.remove(response)
    best_aic = float('inf')
    best_model = None
    all_results = []
    columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)', 'No. Observations',
               'F-statistic', 'F-statistic p-value', 'Intercept', 'Intercept_p-value', 'Std_Intercept', 'Mallows_Cp',
               'Omnibus', 'Prob(Omnibus)', 'Skew', 'Kurtosis', 'Durbin-Watson', 'Jarque-Bera', 'Prob(JB)'] + \
              [f'Std_Coeff_{pred}' for pred in predictors] + \
              [f'Coeff_{pred}' for pred in predictors] + \
              predictors + [f'p-value_{pred}' for pred in predictors] + \
              [f'R2_{pred}' for pred in predictors] + \
              [f'VIF_{pred}' for pred in predictors] + \
              [f'Tolerance_{pred}' for pred in predictors] + \
              ['Shapiro_Wilk', 'KS_Test', 'Anderson_Darling', 'Breusch-Pagan_LM', 'Breusch-Pagan_p-value']

    conflicting_features = ['용적률산정연면적', '연면적']
    for k in range(1, len(predictors) + 1):
        for combo in combinations(predictors, k):
            if sum(feature in combo for feature in conflicting_features) > 1:
                continue
            if '층수' in combo and '지하층수' in combo:
                continue
            if '층수' in combo and '지상층수' in combo:
                continue
            formula = "{} ~ {}".format(response, ' + '.join(combo))
            model = sm.OLS.from_formula(formula, data).fit()

            # 표준화 회귀계수 계산
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[list(combo) + [response]])
            scaled_df = pd.DataFrame(scaled_data, columns=list(combo) + [response])
            std_model = sm.OLS(scaled_df[response], sm.add_constant(scaled_df[list(combo)])).fit()
            standardized_coeffs = std_model.params
            p_values = model.pvalues

            if all(p_values < 0.05):
                mse_full = sm.OLS(data[response], sm.add_constant(data[predictors])).fit().mse_resid
                cp = (model.ssr / mse_full) - (len(data) - 2 * (len(combo) + 1))

                # Omnibus 테스트 및 결과 추출
                omni_test = sm.stats.omni_normtest(model.resid)

                result = {
                    'Combination': ' + '.join(combo),
                    'Num_Predictors': len(combo),
                    'AIC': round(model.aic, 4),
                    'BIC': round(model.bic, 4),
                    'R-squared': round(model.rsquared, 4),
                    'Adj. R-squared': round(model.rsquared_adj, 4),
                    'Prob (F-statistic)': round(model.f_pvalue, 5),
                    'No. Observations': int(model.nobs),
                    'F-statistic': round(model.fvalue, 4),
                    'F-statistic p-value': round(model.f_pvalue, 4),
                    'Intercept': round(model.params['Intercept'], 4) if 'Intercept' in model.params else float('nan'),
                    'Intercept_p-value': round(p_values['Intercept'], 4) if 'Intercept' in p_values else float('nan'),
                    'Std_Intercept': round(standardized_coeffs[0], 4) if len(standardized_coeffs) > 0 else float('nan'),
                    'Mallows_Cp': round(cp, 4),
                    'Omnibus': round(omni_test[0], 4), # Omnibus 통계량
                    'Prob(Omnibus)': round(omni_test[1], 4), # Omnibus p-value
                    'Skew': round(model.resid.skew(), 4),
                    'Kurtosis': round(model.resid.kurtosis(), 4),
                    'Durbin-Watson': round(sm.stats.durbin_watson(model.resid), 4),
                    'Jarque-Bera': round(jarque_bera(model.resid)[0], 4),
                    'Prob(JB)': round(jarque_bera(model.resid)[1], 4)
                }

                for predictor in predictors:
                    if predictor in combo:
                        result[f'Std_Coeff_{predictor}'] = round(standardized_coeffs.get(predictor, float('nan')), 4)
                        result[f'Coeff_{predictor}'] = round(model.params.get(predictor, float('nan')), 4)
                        p_value = model.pvalues.get(predictor, float('nan'))
                        result[predictor] = round(model.params.get(predictor, float('nan')), 4)
                        result[f'p-value_{predictor}'] = round(p_value, 4) if pd.notna(p_value) else float('nan')
                    else:
                        result[f'Std_Coeff_{predictor}'] = float('nan')
                        result[f'Coeff_{predictor}'] = float('nan')
                        result[predictor] = float('nan')
                        result[f'p-value_{predictor}'] = float('nan')

                r2_values, vif_values, tolerance_values = calculate_individual_r2_vif(data, list(combo))
                for predictor in predictors:
                    result[f'R2_{predictor}'] = r2_values.get(predictor, float('nan'))
                    result[f'VIF_{predictor}'] = vif_values.get(predictor, float('nan'))
                    result[f'Tolerance_{predictor}'] = tolerance_values.get(predictor, float('nan'))

                # 정규성 테스트
                shapiro_test = shapiro(model.resid)
                ks_test = kstest(model.resid, 'norm')
                anderson_test = anderson(model.resid)
                jarque_bera_test = jarque_bera(model.resid)

                result['Shapiro_Wilk'] = round(shapiro_test.pvalue, 4)
                result['KS_Test'] = round(ks_test.pvalue, 4)
                result['Anderson_Darling'] = round(anderson_test.statistic, 4)
                result['Jarque_Bera'] = round(jarque_bera_test[1], 4)
                bp_test = het_breuschpagan(model.resid, model.model.exog)
                result['Breusch-Pagan_LM'] = round(bp_test[0], 4)
                result['Breusch-Pagan_p-value'] = round(bp_test[1], 4)

                all_results.append(result)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = result

    return best_model, pd.DataFrame(all_results, columns=columns)

# 데이터 준비 및 최적 모델 탐색
df_new = multiple_regression(hos_1)
response_variable = 'USE_QTY_kWh'
best_model, all_results_df = get_best_model(df_new, response_variable)

# 결과 정리 및 파일 저장
predictors = df_new.columns.tolist()
predictors.remove(response_variable)
presence_columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC','Mallows_Cp', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)',
                    'No. Observations', 'F-statistic', 'F-statistic p-value', 'Intercept'] + \
                   [f'Coeff_{pred}' for pred in predictors] + ['Std_Intercept'] + [f'Std_Coeff_{pred}' for pred in predictors] + \
                   ['Intercept_p-value'] + [f'p-value_{pred}' for pred in predictors] + [f'R2_{pred}' for pred in predictors] + [f'VIF_{pred}' for pred in predictors] + \
                   [f'Tolerance_{pred}' for pred in predictors] + ['Omnibus', 'Prob(Omnibus)', 'Skew', 'Kurtosis', 'Durbin-Watson',
                   'Jarque-Bera', 'Prob(JB)', 'Shapiro_Wilk', 'KS_Test', 'Anderson_Darling', 'Breusch-Pagan_LM', 'Breusch-Pagan_p-value']

all_results_df = all_results_df[presence_columns]

# 예측변수의 존재 여부를 표시
for predictor in predictors:
    all_results_df[predictor] = all_results_df['Combination'].apply(lambda x: 0 if predictor in x.split(' + ') else '')
presence_columns = [pred for pred in predictors] + presence_columns
all_results_df = all_results_df[presence_columns]
# 중복된 presence 컬럼 제거
all_results_df = all_results_df.loc[:, ~all_results_df.columns.duplicated()]
all_results_df = all_results_df.drop_duplicates()
all_results_df['AIC_Rank'] = all_results_df['AIC'].rank(ascending=True)
all_results_df['BIC_Rank'] = all_results_df['BIC'].rank(ascending=True)
all_results_df['Mallows_Cp_Rank'] = all_results_df['Mallows_Cp'].rank(ascending=True)

# 결과 파일 경로 설정 및 저장
dir_path = r"C:\Users\user\Desktop"

dir_path = os.path.join(dir_path, '다중회귀결과')
# output_file = os.path.join(dir_path, f"종합병원_IQR_sqrt_{str(year)}.xlsx")
output_file = os.path.join(dir_path, f"{A}.xlsx")

all_results_df.to_excel(output_file, index=False)

print("모든 조합 결과가 저장되었습니다:", output_file)
print("최적 모델(AIC 기준):")
print(best_model)
