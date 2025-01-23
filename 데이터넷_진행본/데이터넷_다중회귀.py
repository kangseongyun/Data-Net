#
# ############################ 정규화0 다중회귀(0.05필터x) ############################################################################
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from itertools import combinations
# from scipy.stats import pearsonr, shapiro, kstest, anderson, jarque_bera
# from sklearn.preprocessing import StandardScaler
# from statsmodels.stats.diagnostic import het_breuschpagan
#
# # 데이터 로드 및 필터링
# dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
# filename = "데이터넷_1_01_시설정보.xlsx"
# file_path = os.path.join(dir_path, filename)
# df_merge_result = pd.read_excel(file_path)
# print("# of PK3 : ", df_merge_result['매칭표제부PK'].nunique())
# print('# of data : ', df_merge_result.shape[0])
#
# filtered_df = df_merge_result[df_merge_result['주용도(의료시설) 비율(%)'] >= 90]
# print("# of PK3 : ", filtered_df['매칭표제부PK'].nunique())
# print('# of data : ', filtered_df.shape[0])
#
# # 새로운 데이터프레임 생성
# df_new = pd.DataFrame()
# df_new['승강기수'] = filtered_df['비상용승강기수'] + filtered_df['승용승강기수']  # all
# df_new['의사수'] = filtered_df['총의사수']  # all
# df_new['병상수'] = filtered_df['총병상수']  # all
# df_new['용적률산정연면적'] = filtered_df['용적률산정연면적(㎡)']
# df_new['대지면적'] = filtered_df['대지면적(㎡)']  # all
# df_new['연면적'] = filtered_df['연면적(㎡)']  # all
# df_new['지하층수'] = filtered_df['지하층수']  # all
# df_new['지상층수'] = filtered_df['지상층수']  # all
# df_new['층수'] = df_new['지상층수'] + df_new['지하층수']
# # df_new['주용도비율'] = filtered_df['주용도(의료시설) 비율(%)']
# # corr = df_new.corr(method='pearson')
# # file_path = os.path.join(dir_path, '상관계수.xlsx')
# # corr.to_excel(file_path)
# # df_new['USE_QTY_kWh'] = df_new['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)
# df_new['USE_QTY_kWh'] = filtered_df['USE_QTY_kWh']
#
# print('검토')
#
# print('# of data : ', df_new.shape[0])
# print(' ')
# # 의사수 컬럼의 결측치 제거
# df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
#
# print('# of data : ', df_new.shape[0])
# print(' ')
#
#
#
#
# # 연면적과 USE_QTY_kWh 산점도
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=df_new['연면적'], y=df_new['USE_QTY_kWh'])
# sns.scatterplot(x=df_new['의사수'], y=df_new['USE_QTY_kWh'])
# plt.xlabel('의사수 (log transformed)')
# plt.ylabel('USE_QTY_kWh')
# plt.show()
#
#
#
#
# output_file = os.path.join(dir_path, "test.xlsx")
# df_new.to_excel(output_file, index=True)
#
#
#
#
#
# results = {}
# for column in df_new.columns:
#     if column != 'USE_QTY_kWh':
#         corr, p_value = pearsonr(df_new[column], df_new['USE_QTY_kWh'])
#         results[column] = {'Correlation': round(corr, 4), 'p-Value': round(p_value, 4)}
#
# file_path = os.path.join(dir_path, 'corr_result.xlsx')
# results_df = pd.DataFrame(results)
# results_df.to_excel(file_path, index=True)
# print(results_df)
#
#
# def calculate_individual_r2_vif(data, predictors):
#     r2_values = {}
#     vif_values = {}
#     tolerance_values = {}
#     for predictor in predictors:
#         other_predictors = [p for p in predictors if p != predictor]
#         if len(other_predictors) > 0:
#             formula = "{} ~ {}".format(predictor, ' + '.join(other_predictors))
#             model = sm.OLS.from_formula(formula, data).fit()
#             r2 = model.rsquared
#             r2_values[predictor] = round(r2, 4)
#             tolerance = 1 - r2
#             tolerance_values[predictor] = round(tolerance, 4)
#             vif_values[predictor] = round(1 / tolerance, 4) if tolerance > 0 else float('inf')
#         else:
#             r2_values[predictor] = float('nan')
#             vif_values[predictor] = float('nan')
#             tolerance_values[predictor] = float('nan')
#     return r2_values, vif_values, tolerance_values
#
#
# def get_best_model(data, response):
#     predictors = data.columns.tolist()
#     predictors.remove(response)
#     best_aic = float('inf')
#     best_model = None
#     all_results = []
#     columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)', 'No. Observations',
#                'F-statistic', 'F-statistic p-value', 'Intercept', 'Intercept_p-value', 'Std_Intercept', 'Durbin-Watson', 'Mallows_Cp'] + \
#               [f'Std_Coeff_{pred}' for pred in predictors] + \
#               [f'Coeff_{pred}' for pred in predictors] + \
#               predictors + [f'p-value_{pred}' for pred in predictors] + \
#               [f'R2_{pred}' for pred in predictors] + \
#               [f'VIF_{pred}' for pred in predictors] + \
#               [f'Tolerance_{pred}' for pred in predictors] + \
#               ['Shapiro_Wilk', 'KS_Test', 'Anderson_Darling', 'Jarque_Bera', 'Breusch-Pagan_LM', 'Breusch-Pagan_p-value']
#
#     conflicting_features = ['용적률산정연면적', '연면적']
#     for k in range(1, len(predictors) + 1):
#         for combo in combinations(predictors, k):
#             if sum(feature in combo for feature in conflicting_features) > 1:
#                 continue
#             if '층수' in combo and '지하층수' in combo:
#                 continue
#             if '층수' in combo and '지상층수' in combo:
#                 continue
#             formula = "{} ~ {}".format(response, ' + '.join(combo))
#             model = sm.OLS.from_formula(formula, data).fit()
#             print(model.summary())
#
#             # 표준화 회귀계수 계산
#             scaler = StandardScaler()
#             scaled_data = scaler.fit_transform(data[list(combo) + [response]])
#             scaled_df = pd.DataFrame(scaled_data, columns=list(combo) + [response])
#             std_model = sm.OLS(scaled_df[response], sm.add_constant(scaled_df[list(combo)])).fit()
#             standardized_coeffs = std_model.params
#             print(std_model.summary())
#             p_values = model.pvalues
#
#             if all(p_values < 0.05):
#
#                 mse_full = sm.OLS(data[response], sm.add_constant(data[predictors])).fit().mse_resid
#                 cp = (model.ssr / mse_full) - (len(data) - 2 * (len(combo) + 1))
#
#                 result = {
#                     'Combination': ' + '.join(combo),
#                     'Num_Predictors': len(combo),  # 추가: 조합의 개수
#                     'AIC': round(model.aic, 4),
#                     'BIC': round(model.bic, 4),
#                     'R-squared': round(model.rsquared, 4),
#                     'Adj. R-squared': round(model.rsquared_adj, 4),
#                     'Prob (F-statistic)': round(model.f_pvalue, 5),
#                     'No. Observations': int(model.nobs),
#                     'F-statistic': round(model.fvalue, 4),
#                     'F-statistic p-value': round(model.f_pvalue, 4),
#                     'Intercept': round(model.params['Intercept'], 4) if 'Intercept' in model.params else float('nan'),
#                     'Intercept_p-value': round(p_values['Intercept'], 4) if 'Intercept' in p_values else float('nan'),
#                     'Std_Intercept': round(standardized_coeffs[0], 4) if len(standardized_coeffs) > 0 else float('nan'),
#                     'Durbin-Watson': round(sm.stats.durbin_watson(model.resid), 4),  # 독립성 값 추가
#                     'Mallows_Cp': round(cp, 4)
#                 }
#                 for predictor in predictors:
#                     if predictor in combo:
#                         result[f'Std_Coeff_{predictor}'] = round(standardized_coeffs.get(predictor, float('nan')), 4)
#                         result[f'Coeff_{predictor}'] = round(model.params.get(predictor, float('nan')), 4)
#                         p_value = model.pvalues.get(predictor, float('nan'))
#                         result[predictor] = round(model.params.get(predictor, float('nan')), 4)
#                         result[f'p-value_{predictor}'] = round(p_value, 4) if pd.notna(p_value) else float('nan')
#                     else:
#                         result[f'Std_Coeff_{predictor}'] = float('nan')
#                         result[f'Coeff_{predictor}'] = float('nan')
#                         result[predictor] = float('nan')
#                         result[f'p-value_{predictor}'] = float('nan')
#
#                 r2_values, vif_values, tolerance_values = calculate_individual_r2_vif(data, list(combo))
#                 for predictor in predictors:
#                     result[f'R2_{predictor}'] = r2_values.get(predictor, float('nan'))
#                     result[f'VIF_{predictor}'] = vif_values.get(predictor, float('nan'))
#                     result[f'Tolerance_{predictor}'] = tolerance_values.get(predictor, float('nan'))
#
#                 # 정규성 테스트
#                 shapiro_test = shapiro(model.resid)
#                 ks_test = kstest(model.resid, 'norm')
#                 anderson_test = anderson(model.resid)
#                 jarque_bera_test = jarque_bera(model.resid)
#
#                 result['Shapiro_Wilk'] = round(shapiro_test.pvalue, 4)
#                 result['KS_Test'] = round(ks_test.pvalue, 4)
#                 result['Anderson_Darling'] = round(anderson_test.statistic, 4)  # Anderson-Darling은 p-value 대신 통계량을 사용합니다
#                 result['Jarque_Bera'] = round(jarque_bera_test[1], 4)  # p-value
#                 bp_test = het_breuschpagan(model.resid, model.model.exog)
#                 result['Breusch-Pagan_LM'] = round(bp_test[0], 4)
#                 result['Breusch-Pagan_p-value'] = round(bp_test[1], 4)
#                 all_results.append(result)
#                 if model.aic < best_aic:
#                     best_aic = model.aic
#                     best_model = result
#
#     return best_model, pd.DataFrame(all_results, columns=columns)
#
#
# response_variable = 'USE_QTY_kWh'
# best_model, all_results_df = get_best_model(df_new, response_variable)
#
# # Reorder columns to have the predictors' presence first
# predictors = df_new.columns.tolist()
# predictors.remove(response_variable)
# presence_columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC','Mallows_Cp', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)',
#                     'No. Observations', 'F-statistic', 'F-statistic p-value', 'Durbin-Watson', 'Intercept'] + \
#                    [f'Coeff_{pred}' for pred in predictors] + ['Std_Intercept'] + [f'Std_Coeff_{pred}' for pred in predictors] + \
#                    ['Intercept_p-value']+[f'p-value_{pred}' for pred in predictors] + [f'R2_{pred}' for pred in predictors] + [f'VIF_{pred}' for pred in predictors] + \
#                    [f'Tolerance_{pred}' for pred in predictors] + ['Shapiro_Wilk', 'KS_Test', 'Anderson_Darling', 'Jarque_Bera', 'Breusch-Pagan_LM', 'Breusch-Pagan_p-value']
#
# all_results_df = all_results_df[presence_columns]
#
# # Adding presence indicators
# for predictor in predictors:
#     all_results_df[predictor] = all_results_df['Combination'].apply(lambda x: 0 if predictor in x.split(' + ') else '')
#
# # Reordering columns to have presence indicators first
# presence_columns = [pred for pred in predictors] + presence_columns
# all_results_df = all_results_df[presence_columns]
#
# # Drop duplicate presence columns
# all_results_df = all_results_df.loc[:, ~all_results_df.columns.duplicated()]
# all_results_df = all_results_df.drop_duplicates()
# all_results_df['AIC_Rank'] = all_results_df['AIC'].rank(ascending=True)
# all_results_df['BIC_Rank'] = all_results_df['BIC'].rank(ascending=True)
# all_results_df['Mallows_Cp_Rank'] = all_results_df['Mallows_Cp'].rank(ascending=True)
#
# output_file = os.path.join(dir_path, "all_combinations_results_with_R2_VIF_Tolerance.xlsx")
# all_results_df.to_excel(output_file, index=False)
#
# print("All combinations results with R2, VIF, Tolerance values, and Durbin-Watson statistics have been saved to:", output_file)
#
# print("Best model with lowest AIC:")
# print(best_model)


############################ 정규화0 다중회귀(0.05필터x) ############################################################################
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import pearsonr, shapiro, kstest, anderson, jarque_bera, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan

# 데이터 로드 및 필터링
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
df_merge_result = pd.read_excel(file_path)
print("# of PK3 : ", df_merge_result['매칭표제부PK'].nunique())
print('# of data : ', df_merge_result.shape[0])

filtered_df = df_merge_result[df_merge_result['주용도(의료시설) 비율(%)'] >= 90]
print("# of PK3 : ", filtered_df['매칭표제부PK'].nunique())
print('# of data : ', filtered_df.shape[0])
filtered_df = filtered_df[filtered_df['year_use'] == 2022]

# 새로운 데이터프레임 생성
df_new = pd.DataFrame()
df_new['연면적'] = filtered_df['연면적(㎡)']  # all
df_new['용적률산정연면적'] = filtered_df['용적률산정연면적(㎡)']
df_new['대지면적'] = filtered_df['대지면적(㎡)']  # all
df_new['층수'] = filtered_df['지상층수'] + filtered_df['지하층수']
df_new['지하층수'] = filtered_df['지하층수']  # all
df_new['지상층수'] = filtered_df['지상층수']  # all
df_new['승강기수'] = filtered_df['비상용승강기수'] + filtered_df['승용승강기수']  # all
df_new['의사수'] = filtered_df['총의사수']  # all
df_new['병상수'] = filtered_df['총병상수']  # all


# df_new['주용도비율'] = filtered_df['주용도(의료시설) 비율(%)']

df_new['USE_QTY_kWh'] = filtered_df['USE_QTY_kWh']


# # #
# # 1사분위수(Q1)와 3사분위수(Q3) 계산
# Q1 = df_new['USE_QTY_kWh'].quantile(0.25)
# Q3 = df_new['USE_QTY_kWh'].quantile(0.75)
#
# # IQR 계산
# IQR = Q3 - Q1
#
# # 이상치 경계값 계산
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # 이상치 제거
# df_new = df_new[(df_new['USE_QTY_kWh'] >= lower_bound) & (df_new['USE_QTY_kWh'] <= upper_bound)]
#



# df_new['USE_QTY_kWh'] = df_new['USE_QTY_kWh'].apply(lambda x: np.sqrt(x) if x > 0 else 0)







print('검토')

print('# of data : ', df_new.shape[0])
print(' ')
# 의사수 컬럼의 결측치 제거
df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])

print('# of data : ', df_new.shape[0])
print(' ')
# df_new['EUI']=df_new['USE_QTY_kWh']/df_new['연면적']

# df_new=df_new.drop(columns=['USE_QTY_kWh'])
#
# # 연면적과 USE_QTY_kWh 산점도
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=df_new['연면적'], y=df_new['USE_QTY_kWh'])
# sns.scatterplot(x=df_new['의사수'], y=df_new['USE_QTY_kWh'])
# plt.xlabel('의사수 (log transformed)')
# plt.ylabel('USE_QTY_kWh')
# plt.show()


corr = df_new.corr(method='pearson')
file_path = os.path.join(dir_path, '상관계수.xlsx')
corr.to_excel(file_path)

output_file = os.path.join(dir_path, "test.xlsx")
df_new.to_excel(output_file, index=True)





results = {}
for column in df_new.columns:
    if column != 'USE_QTY_kWh':
        corr, p_value = pearsonr(df_new[column], df_new['USE_QTY_kWh'])
        results[column] = {'Correlation': round(corr, 4), 'p-Value': round(p_value, 4)}

file_path = os.path.join(dir_path, 'corr_result.xlsx')
results_df = pd.DataFrame(results)
results_df.to_excel(file_path, index=True)
print(results_df)



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

def get_best_model(data, response):
    predictors = data.columns.tolist()
    predictors.remove(response)
    best_aic = float('inf')
    best_model = None
    all_results = []
    columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)',
               'No. Observations', 'F-statistic', 'F-statistic p-value', 'Intercept', 'Intercept_p-value',
               'Std_Intercept', 'Durbin-Watson', 'Mallows_Cp', 'Skew', 'Kurtosis'] + \
              [f'Std_Coeff_{pred}' for pred in predictors] + \
              [f'Coeff_{pred}' for pred in predictors] + \
              predictors + [f'p-value_{pred}' for pred in predictors] + \
              [f'R2_{pred}' for pred in predictors] + \
              [f'VIF_{pred}' for pred in predictors] + \
              [f'Tolerance_{pred}' for pred in predictors] + \
              ['Shapiro_Wilk', 'KS_Test', 'Anderson_Darling', 'Jarque_Bera', 'Breusch-Pagan_LM',
               'Breusch-Pagan_p-value']

    # conflicting_features = ['용적률산정연면적', '연면적']
    for k in range(1, len(predictors) + 1):
        for combo in combinations(predictors, k):
            # if sum(feature in combo for feature in conflicting_features) > 1:
            #     continue
            # if '층수' in combo and '지하층수' in combo:
            #     continue
            # if '층수' in combo and '지상층수' in combo:
            #     continue
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
                'Durbin-Watson': round(sm.stats.durbin_watson(model.resid), 4),
                'Mallows_Cp': round(cp, 4),
                'Skew': round(model_skew, 4),
                'Kurtosis': round(model_kurtosis, 4)
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

            # Normality and heteroscedasticity tests
            shapiro_test = shapiro(model.resid)
            ks_test = kstest(model.resid, 'norm')
            anderson_test = anderson(model.resid)
            jarque_bera_test = jarque_bera(model.resid)
            bp_test = het_breuschpagan(model.resid, model.model.exog)

            result['Shapiro_Wilk'] = round(shapiro_test.pvalue, 4)
            result['KS_Test'] = round(ks_test.pvalue, 4)
            result['Anderson_Darling'] = round(anderson_test.statistic, 4)
            result['Jarque_Bera'] = round(jarque_bera_test[1], 4)
            result['Breusch-Pagan_LM'] = round(bp_test[0], 4)
            result['Breusch-Pagan_p-value'] = round(bp_test[1], 4)

            all_results.append(result)
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = result

    return best_model, pd.DataFrame(all_results, columns=columns)

response_variable = 'USE_QTY_kWh'
best_model, all_results_df = get_best_model(df_new, response_variable)

# Reorder columns to have the predictors' presence first
predictors = df_new.columns.tolist()
predictors.remove(response_variable)
presence_columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC','Mallows_Cp', 'R-squared', 'Adj. R-squared', 'Prob (F-statistic)',
                    'No. Observations', 'F-statistic', 'F-statistic p-value', 'Durbin-Watson', 'Intercept'] + \
                   [f'Coeff_{pred}' for pred in predictors] + ['Std_Intercept'] + [f'Std_Coeff_{pred}' for pred in predictors] + \
                   ['Intercept_p-value']+[f'p-value_{pred}' for pred in predictors] + [f'R2_{pred}' for pred in predictors] + [f'VIF_{pred}' for pred in predictors] + \
                   [f'Tolerance_{pred}' for pred in predictors] + ['Skew', 'Kurtosis', 'Shapiro_Wilk', 'KS_Test',
                   'Anderson_Darling', 'Jarque_Bera', 'Breusch-Pagan_LM', 'Breusch-Pagan_p-value']

all_results_df = all_results_df[presence_columns]

# Adding presence indicators
for predictor in predictors:
    all_results_df[predictor] = all_results_df['Combination'].apply(lambda x: 0 if predictor in x.split(' + ') else '')

# Reordering columns to have presence indicators first
presence_columns = [pred for pred in predictors] + presence_columns
all_results_df = all_results_df[presence_columns]

# Drop duplicate presence columns
all_results_df = all_results_df.loc[:, ~all_results_df.columns.duplicated()]
all_results_df = all_results_df.drop_duplicates()
all_results_df['AIC_Rank'] = all_results_df['AIC'].rank(ascending=True)
all_results_df['BIC_Rank'] = all_results_df['BIC'].rank(ascending=True)
all_results_df['Mallows_Cp_Rank'] = all_results_df['Mallows_Cp'].rank(ascending=True)

# output_file = os.path.join(dir_path, "IQR데이터.xlsx")
# output_file = os.path.join(dir_path, "기본데이터.xlsx")
# all_results_df.to_excel(output_file, index=False)
#
# print("All combinations results with R2, VIF, Tolerance values, and Durbin-Watson statistics have been saved to:", output_file)
#
# print("Best model with lowest AIC:")
# print(best_model)