import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from matplotlib import pyplot as plt
from scipy.stats import shapiro, kstest, anderson, jarque_bera, normaltest, boxcox, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')
# 데이터 로드
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
data = pd.read_excel(file_path)

year=2022

data = data[data['year_use'] == year]

Group_1 = data[data['종별코드명'].isin(['종합병원'])]

Group_2 = data[data['종별코드명'].isin(['병원','치과병원'])]

Group_3 = data[data['종별코드명'].isin(["한방병원",'정신병원','치과병원'])]



dir_path_porder1 = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본\그룹화"



#
# dir_path_porder3 = "Group_1"
# Analysis_target=Group_1

dir_path_porder3 = "Group_2"
Analysis_target=Group_2
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


    return hos

Analysis_target=transform_hos(Analysis_target, dir_path_porder_method)



# dir_path_porder_excel = os.path.join(dir_path_porder6, f"{dir_path_porder3}_{dir_path_porder_method}_Total.xlsx")

str_year = str(year)
dir_path_porder_excel = os.path.join(dir_path_porder1, f"{dir_path_porder3}_{dir_path_porder_method}_{str_year}.xlsx")
data = Analysis_target[Analysis_target['year_use'].isin([year])]



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

data_n = data_n[(data_n['대지면적'] > 0) & (data_n['의사수'] > 0) & (data_n['지상층수'] > 0)]
print('# of data1 : ', len(data_n))

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

# Glejser 검정 함수 구현
def glejser_test(residuals, exog):
    abs_resid = np.abs(residuals)
    exog = sm.add_constant(exog)
    glejser_model = sm.OLS(abs_resid, exog).fit()
    return glejser_model.f_pvalue, glejser_model.fvalue


# 정규성 검사를 위한 함수 추가
def normality_tests(residuals):
    shapiro_test = shapiro(residuals)
    anderson_test = anderson(residuals)
    jarque_bera_test = jarque_bera(residuals)
    normal_test = normaltest(residuals)
    kstest_result = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))

    normality_results = {
        'Shapiro-Wilk': round(shapiro_test.pvalue, 4),
        'Anderson-Darling': round(anderson_test.statistic, 4),
        'Jarque-Bera': round(jarque_bera_test[1], 4),
        'Normal_Test': round(normal_test.pvalue, 4),
        'KS_Test': round(kstest_result.pvalue, 4)
    }

    return normality_results

# 최적 모델 탐색 함수 내에서 정상성 검정 결과를 포함하도록 수정
def get_best_model(data, response):
    predictors = data.columns.tolist()
    predictors.remove(response)
    all_results = []
    columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)', 'No. Observations',
               'F-statistic', 'F-statistic p-value', 'Intercept', 'Intercept_p-value', 'Std_Intercept', 'Mallows_Cp',
               'Omnibus', 'Prob(Omnibus)', 'Skew', 'Kurtosis', 'Durbin-Watson', 'Jarque-Bera', 'Prob(JB)',
               'Shapiro_Wilk', 'KS_Test', 'Anderson_Darling', 'Normal_Test', 'Breusch-Pagan_LM', 'Breusch-Pagan_p-value',
               'White_Test', 'White_p-value', 'Glejser_Test', 'Glejser_p-value'] + \
              [f'Std_Coeff_{pred}' for pred in predictors] + \
              [f'Coeff_{pred}' for pred in predictors] + \
              predictors + [f'p-value_{pred}' for pred in predictors] + \
              [f'R2_{pred}' for pred in predictors] + \
              [f'VIF_{pred}' for pred in predictors] + \
              [f'Tolerance_{pred}' for pred in predictors]

    for k in range(1, len(predictors) + 1):
        for combo in combinations(predictors, k):
            formula = "{} ~ {}".format(response, ' + '.join(combo))
            model = sm.OLS.from_formula(formula, data).fit()

            # 표준화 회귀계수 계산
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[list(combo) + [response]])
            scaled_df = pd.DataFrame(scaled_data, columns=list(combo) + [response])
            std_model = sm.OLS(scaled_df[response], sm.add_constant(scaled_df[list(combo)])).fit()
            standardized_coeffs = std_model.params
            p_values = model.pvalues

            mse_full = sm.OLS(data[response], sm.add_constant(data[predictors])).fit().mse_resid
            cp = (model.ssr / mse_full) - (len(data) - 2 * (len(combo) + 1))

            # Omnibus 테스트 및 결과 추출
            omni_test = sm.stats.omni_normtest(model.resid)

            # 정규성 테스트 결과 추가
            norm_tests = normality_tests(model.resid)
            # Skewness and Kurtosis
            model_skew = skew(model.resid)
            model_kurtosis = kurtosis(model.resid)
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
                'Omnibus': round(omni_test[0], 4),
                'Prob(Omnibus)': round(omni_test[1], 4),
                'Skew': round(model_skew, 4),
                'Kurtosis': round(model_kurtosis, 4),
                'Durbin-Watson': round(sm.stats.durbin_watson(model.resid), 4),
                'Jarque-Bera': round(jarque_bera(model.resid)[0], 4),
                'Prob(JB)': round(jarque_bera(model.resid)[1], 4),
                'Shapiro_Wilk': norm_tests['Shapiro-Wilk'],
                'KS_Test': norm_tests['KS_Test'],
                'Anderson_Darling': norm_tests['Anderson-Darling'],
                'Normal_Test': norm_tests['Normal_Test'],
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

            # 등분산성 테스트 (Breusch-Pagan)
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            result['Breusch-Pagan_LM'] = round(bp_test[0], 4)
            result['Breusch-Pagan_p-value'] = round(bp_test[1], 4)



            # Glejser 검정
            glejser_pvalue, glejser_fvalue = glejser_test(model.resid, model.model.exog)
            result['Glejser_Test'] = round(glejser_fvalue, 4)
            result['Glejser_p-value'] = round(glejser_pvalue, 4)

            all_results.append(result)

    return pd.DataFrame(all_results, columns=columns)



# 데이터 준비 및 최적 모델 탐색
response_variable = 'USE_QTY_kWh'
all_results_df = get_best_model(data_n, response_variable)

# 결과 정리 및 파일 저장
predictors = data_n.columns.tolist()
predictors.remove(response_variable)
presence_columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC', 'Mallows_Cp', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)',
                    'No. Observations', 'F-statistic', 'F-statistic p-value', 'Intercept'] + \
                   ['Skew', 'Kurtosis', 'Omnibus', 'Prob(Omnibus)', 'Jarque-Bera', 'Prob(JB)', 'Shapiro_Wilk',
                    'KS_Test', 'Anderson_Darling', 'Normal_Test',  # 정규성 관련
                    'Breusch-Pagan_LM', 'Breusch-Pagan_p-value',  # 등분산성 관련
                    'White_Test', 'White_p-value', 'Glejser_Test', 'Glejser_p-value',  # 추가된 등분산성 관련
                    'Durbin-Watson']  + \
                   [f'Coeff_{pred}' for pred in predictors] + ['Std_Intercept'] + [f'Std_Coeff_{pred}' for pred in predictors] + \
                   ['Intercept_p-value'] + [f'p-value_{pred}' for pred in predictors] + [f'R2_{pred}' for pred in predictors] + [f'VIF_{pred}' for pred in predictors] + \
                   [f'Tolerance_{pred}' for pred in predictors]


# 존재하는 열만 필터링하여 선택
presence_columns = [col for col in presence_columns if col in all_results_df.columns]

# 필터링된 열들로 데이터프레임 생성
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

all_results_df.to_excel(dir_path_porder_excel, index=False)
print("모든 조합 결과가 저장되었습니다:", dir_path_porder_excel)
