import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from scipy.stats import boxcox
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')

dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
q1 = pd.read_excel(file_path)

# 대상 병원 필터
hos_1 = q1[q1['종별코드명'].isin(['병원', '치과병원', '한방병원', '요양병원', '정신병원'])]
hos_2 = q1[q1['종별코드명'].isin(['종합병원'])]
# hos_2 = hos_2[hos_2['year_use'].isin([2018, 2019, 2022])]
hos_2 = hos_2[hos_2['year_use'].isin([2020, 2021])]


def change_set(A):
    df_new = pd.DataFrame()

    df_new['승강기수'] = A['비상용승강기수'] + A['승용승강기수']  # all
    df_new['의사수'] = A['총의사수']  # all
    df_new['병상수'] = A['총병상수']  # all
    df_new['용적률산정연면적'] = A['용적률산정연면적(㎡)']
    df_new['대지면적'] = A['대지면적(㎡)']  # all
    df_new['연면적'] = A['연면적(㎡)']  # all
    df_new['지하층수'] = A['지하층수']  # all
    df_new['지상층수'] = A['지상층수']  # all
    df_new['층수'] = df_new['지상층수'] + df_new['지하층수']
    df_new['USE_QTY_kWh'] = A['USE_QTY_kWh']
    df_new = df_new.sort_values(by='USE_QTY_kWh', ascending=True)

    # 1사분위수(Q1)와 3사분위수(Q3) 계산
    Q1 = df_new['USE_QTY_kWh'].quantile(0.25)
    Q3 = df_new['USE_QTY_kWh'].quantile(0.75)

    # IQR 계산
    IQR = Q3 - Q1

    # 이상치 경계값 계산
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치 제거
    df_new = df_new[(df_new['USE_QTY_kWh'] >= lower_bound) & (df_new['USE_QTY_kWh'] <= upper_bound)]

    return df_new


def calculate_vif_and_tolerance(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    return vif_data.set_index("Variable")


def calculate_mallows_cp(model, full_model_sse, full_model_df):
    residual_sum_of_squares = model.ssr
    sigma_squared = full_model_sse / full_model_df
    n = model.nobs
    p = model.df_model + 1  # Including the intercept
    cp = (residual_sum_of_squares / sigma_squared) - (n - 2 * p)
    return cp


def run_regressions_and_export(df, independent_vars, dependent_var):
    results_list = []

    # Box-Cox transformation of dependent variable
    y_bc, lambda_ = boxcox(df[dependent_var])
    print(f"Box-Cox Lambda: {lambda_}")

    # Full model for Mallows' Cp calculation
    X_full = sm.add_constant(df[independent_vars])
    full_model = sm.OLS(y_bc, X_full).fit()
    full_model_sse = full_model.ssr
    full_model_df = full_model.df_resid

    # Iterate over all combinations of independent variables
    for L in range(1, len(independent_vars) + 1):
        for combo in itertools.combinations(independent_vars, L):
            X = df[list(combo)]
            X = sm.add_constant(X)  # Add constant term

            model = sm.OLS(y_bc, X).fit()

            # Skip models with p-value > 0.05 for F-statistic (model's p-value)
            if model.f_pvalue > 0.05:
                continue

            p_values = model.pvalues

            # Calculate VIF and Tolerance for each variable in the model
            vif_df = calculate_vif_and_tolerance(X)

            # Skip models where any variable's p-value > 0.05, VIF > 10, or Tolerance < 0.1 (optional for strict filtering)
            if any(p_values[1:] > 0.05) or any(vif_df["VIF"] > 10):
                continue

            # Standardized coefficients
            standardized_coeffs = model.params / model.bse

            # Calculate Mallows' Cp
            cp = calculate_mallows_cp(model, full_model_sse, full_model_df)

            # Calculate additional statistics
            omnibus = sm.stats.omni_normtest(model.resid)
            jb_test = sm.stats.jarque_bera(model.resid)
            cond_no = np.linalg.cond(X)

            # Initialize the result row
            result_row = {
                'Combination': ' + '.join(combo),
                'Num_Predictors': len(combo),
                'Lambda': lambda_,
                'AIC': round(model.aic, 4),
                'BIC': round(model.bic, 4),
                'R-squared': round(model.rsquared, 4),
                'Adj. R-squared': round(model.rsquared_adj, 4),
                'Prob (F-statistic)': round(model.f_pvalue, 5),
                'No. Observations': int(model.nobs),
                'F-statistic': round(model.fvalue, 4),
                'F-statistic p-value': round(model.f_pvalue, 4),
                'Durbin-Watson': round(sm.stats.durbin_watson(model.resid), 4),
                'Mallows_Cp': round(cp, 4),
                'Omnibus': round(omnibus.statistic, 4),
                'Prob(Omnibus)': round(omnibus.pvalue, 5),
                'Skew': round(pd.Series(model.resid).skew(), 4),
                'Kurtosis': round(pd.Series(model.resid).kurtosis(), 4),
                'Jarque-Bera (JB)': round(jb_test[0], 4),
                'Prob(JB)': round(jb_test[1], 5),
                'Cond. No.': round(cond_no, 4)
            }

            # Add each independent variable's result to the row grouped by metric type
            for var in combo:
                result_row[f'{var}_Coefficient'] = round(model.params[var], 4)
                result_row[f'{var}_Std Err'] = round(model.bse[var], 4)
                result_row[f'{var}_t-value'] = round(model.tvalues[var], 4)
                result_row[f'{var}_P>|t|'] = round(model.pvalues[var], 4)
                result_row[f'{var}_Std_Coefficient'] = round(standardized_coeffs[var],
                                                             4) if var in standardized_coeffs else float('nan')
                result_row[f'{var}_VIF'] = round(vif_df.loc[var, "VIF"], 4) if var in vif_df.index else float('nan')
                result_row[f'{var}_Tolerance'] = round(vif_df.loc[var, "Tolerance"], 4) if var in vif_df.index else float('nan')

            # Add Intercept (constant) values, if present
            if 'const' in model.params:
                result_row['Intercept_Coefficient'] = round(model.params['const'], 4)
                result_row['Intercept_Std Err'] = round(model.bse['const'], 4)
                result_row['Intercept_t-value'] = round(model.tvalues['const'], 4)
                result_row['Intercept_P>|t|'] = round(p_values['const'], 4)
                result_row['Intercept_Std_Coefficient'] = round(standardized_coeffs['const'],
                                                                4) if 'const' in standardized_coeffs else float('nan')

            results_list.append(result_row)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Reorder columns so that metrics are grouped together (_Coefficient, _Std Err, etc.)
    ordered_columns = ['Combination', 'Num_Predictors', 'Lambda', 'AIC', 'BIC', 'R-squared', 'Adj. R-squared',
                       'Prob (F-statistic)', 'No. Observations', 'F-statistic', 'F-statistic p-value',
                       'Durbin-Watson', 'Mallows_Cp', 'Omnibus', 'Prob(Omnibus)', 'Skew', 'Kurtosis',
                       'Jarque-Bera (JB)', 'Prob(JB)', 'Cond. No.']

    metrics = ['_Coefficient', '_Std Err', '_t-value', '_P>|t|', '_Std_Coefficient', '_VIF', '_Tolerance']

    for metric in metrics:
        for col in results_df.columns:
            if metric in col and col not in ordered_columns:
                ordered_columns.append(col)

    results_df = results_df[ordered_columns]

    # Export to Excel
    results_df.to_excel('regression_results_filtered_by_conditions2.xlsx', index=False)


# 데이터 전처리 (이미 전처리 된 데이터프레임 hos_2 사용)
hos_2 = change_set(hos_2)

# 독립 변수와 종속 변수 설정
independent_vars = ['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수']
dependent_var = 'USE_QTY_kWh'

# 회귀 분석 수행 및 결과를 엑셀로 추출
run_regressions_and_export(hos_2, independent_vars, dependent_var)
