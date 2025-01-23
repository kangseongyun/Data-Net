import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import pearsonr, shapiro, kstest, anderson, jarque_bera
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import boxcox
from sklearn.metrics import silhouette_score

# 데이터 로드 및 필터링
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
df_merge_result = pd.read_excel(file_path)
print("# of PK3 : ", df_merge_result['매칭표제부PK'].nunique())
print('# of data : ', df_merge_result.shape[0])

filtered_df = df_merge_result[df_merge_result['주용도(의료시설) 비율(%)'] >= 80]
print("# of PK3 : ", filtered_df['매칭표제부PK'].nunique())
print('# of data : ', filtered_df.shape[0])

# 새로운 데이터프레임 생성
df_new = pd.DataFrame()
df_new['승강기수'] = filtered_df['비상용승강기수'] + filtered_df['승용승강기수']
df_new['의사수'] = filtered_df['총의사수']
df_new['병상수'] = filtered_df['총병상수']
df_new['용적률산정연면적'] = filtered_df['용적률산정연면적(㎡)']
df_new['대지면적'] = filtered_df['대지면적(㎡)']
df_new['연면적'] = filtered_df['연면적(㎡)']

df_new['면적대비승강기수'] = df_new['승강기수'] / df_new['연면적']
df_new['면적대비의사수'] = df_new['의사수'] / df_new['연면적']
df_new['면적대비병상수'] = df_new['병상수'] / df_new['연면적']

df_new['지하층수'] = filtered_df['지하층수']
df_new['지상층수'] = filtered_df['지상층수']
df_new['층수'] = df_new['지상층수'] + df_new['지하층수']

df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
# df_new = np.log(df_new)
df_new['USE_QTY_kWh'] = filtered_df['USE_QTY_kWh']

# 데이터 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_new)


# 엘보우 방법을 사용하여 최적의 군집 개수 찾기
def find_optimal_clusters(data, max_k):
    iters = range(1, max_k + 1)
    sse = []
    silhouette_scores = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

        if k > 1:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(iters, sse, marker='o')
    ax1.set_xlabel('Cluster Centers')
    ax1.set_ylabel('SSE')
    ax1.set_title('Elbow Method For Optimal k')

    ax2.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    ax2.set_xlabel('Cluster Centers')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores For Optimal k')

    plt.show()


find_optimal_clusters(scaled_data, 10)

# 위의 그래프를 보고 최적의 클러스터 수를 결정합니다.
optimal_clusters = 4  # 예: 엘보우와 실루엣 스코어가 최적이 되는 군집 수로 설정

# 최적의 클러스터 개수로 클러스터링 수행
kmeans = KMeans(n_clusters=optimal_clusters, random_state=7, n_init=10)
kmeans.fit(scaled_data)
df_new['Cluster'] = kmeans.labels_

# 클러스터링 결과 시각화
for cluster in df_new['Cluster'].unique():
    cluster_data = df_new[df_new['Cluster'] == cluster]
    plt.scatter(cluster_data['연면적'], cluster_data['USE_QTY_kWh'], label=f'Cluster {cluster}')

plt.xlabel('Gross Floor Area (연면적)')
plt.ylabel('USE_QTY_kWh')
plt.title('Scatter plot of USE_QTY_kWh by Gross Floor Area for each Cluster')
plt.legend()
plt.grid(True)
plt.show()

# 클러스터별 USE_QTY_kWh 통계
cluster_stats = df_new.groupby('Cluster')['연면적'].describe()

# 클러스터별로 데이터프레임 분리
clusters_dfs = [df_new[df_new['Cluster'] == i].drop(columns=['Cluster']) for i in range(optimal_clusters)]

# 결과를 엑셀 파일로 저장
output_file_path = os.path.join(dir_path, '최적_클러스터_결과.xlsx')
with pd.ExcelWriter(output_file_path) as writer:
    df_new.to_excel(writer, sheet_name='전체 데이터', index=False)
    for i, cluster_df in enumerate(clusters_dfs):
        cluster_df.to_excel(writer, sheet_name=f'클러스터 a{i}', index=False)

print("클러스터링 결과가 엑셀 파일로 저장되었습니다.")


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
               'No. Observations',
               'F-statistic', 'F-statistic p-value', 'Intercept', 'Intercept_p-value', 'Std_Intercept',
               'Durbin-Watson'] + \
              [f'Std_Coeff_{pred}' for pred in predictors] + \
              [f'Coeff_{pred}' for pred in predictors] + \
              predictors + [f'p-value_{pred}' for pred in predictors] + \
              [f'R2_{pred}' for pred in predictors] + \
              [f'VIF_{pred}' for pred in predictors] + \
              [f'Tolerance_{pred}' for pred in predictors] + \
              ['Shapiro_Wilk', 'KS_Test', 'Anderson_Darling', 'Jarque_Bera', 'Breusch-Pagan_LM',
               'Breusch-Pagan_p-value']

    conflicting_features = ['용적률산정연면적', '대지면적', '연면적']
    for k in range(1, len(predictors) + 1):
        for combo in combinations(predictors, k):
            if sum(feature in combo for feature in conflicting_features) > 1:
                continue
            if '층수' in combo and '지하층수' in combo:
                continue
            if '층수' in combo and '지상층수' in combo:
                continue
            if '면적대비승강기수' in combo and '승강기수' in combo:
                continue
            if '면적대비의사수' in combo and '의사수' in combo:
                continue
            if '면적대비병상수' in combo and '병상수' in combo:
                continue

            formula = "{} ~ {}".format(response, ' + '.join(combo))
            model = sm.OLS.from_formula(formula, data).fit()
            print(model.summary())

            # 표준화 회귀계수 계산
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[list(combo) + [response]])
            scaled_df = pd.DataFrame(scaled_data, columns=list(combo) + [response])
            std_model = sm.OLS(scaled_df[response], sm.add_constant(scaled_df[list(combo)])).fit()
            standardized_coeffs = std_model.params
            print(std_model.summary())
            p_values = model.pvalues

            if all(p_values < 0.05):
                # if all(p_values < 0.05):
                result = {
                    'Combination': ' + '.join(combo),
                    'Num_Predictors': len(combo),  # 추가: 조합의 개수
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
                    'Durbin-Watson': round(sm.stats.durbin_watson(model.resid), 4)  # 독립성 값 추가
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
                result['Anderson_Darling'] = round(anderson_test.statistic,
                                                   4)  # Anderson-Darling은 p-value 대신 통계량을 사용합니다
                result['Jarque_Bera'] = round(jarque_bera_test[1], 4)  # p-value
                bp_test = het_breuschpagan(model.resid, model.model.exog)
                result['Breusch-Pagan_LM'] = round(bp_test[0], 4)
                result['Breusch-Pagan_p-value'] = round(bp_test[1], 4)
                all_results.append(result)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = result

    return best_model, pd.DataFrame(all_results, columns=columns)


response_variable = 'USE_QTY_kWh'
# 클러스터별 다중회귀 결과 저장
with pd.ExcelWriter(output_file_path) as writer:
    df_new.to_excel(writer, sheet_name='전체 데이터', index=False)
    for i, cluster_df in enumerate(clusters_dfs):
        best_model, all_results_df = get_best_model(cluster_df, response_variable)

        # Reorder columns to have the predictors' presence first
        predictors = cluster_df.columns.tolist()
        predictors.remove(response_variable)

        presence_columns = ['Combination', 'Num_Predictors', 'AIC', 'BIC', 'R-squared', 'Adj. R-squared',
                            'Prob (F-statistic)',
                            'No. Observations', 'F-statistic', 'F-statistic p-value', 'Durbin-Watson', 'Intercept'] + \
                           [f'Coeff_{pred}' for pred in predictors] + ['Std_Intercept'] + [f'Std_Coeff_{pred}' for pred
                                                                                           in predictors] + \
                           ['Intercept_p-value'] + [f'p-value_{pred}' for pred in predictors] + [f'R2_{pred}' for pred
                                                                                                 in predictors] + [
                               f'VIF_{pred}' for pred in predictors] + \
                           [f'Tolerance_{pred}' for pred in predictors] + ['Shapiro_Wilk', 'KS_Test',
                                                                           'Anderson_Darling', 'Jarque_Bera',
                                                                           'Breusch-Pagan_LM', 'Breusch-Pagan_p-value']

        all_results_df = all_results_df[presence_columns]

        # Adding presence indicators
        for predictor in predictors:
            all_results_df[predictor] = all_results_df['Combination'].apply(
                lambda x: 0 if predictor in x.split(' + ') else '')

        # Reordering columns to have presence indicators first
        presence_columns = [pred for pred in predictors] + presence_columns
        all_results_df = all_results_df[presence_columns]

        # Drop duplicate presence columns
        all_results_df = all_results_df.loc[:, ~all_results_df.columns.duplicated()]
        all_results_df = all_results_df.drop_duplicates()

        all_results_df.to_excel(writer, sheet_name=f'클러스터 a{i}_회귀결과', index=False)
        cluster_df.to_excel(writer, sheet_name=f'클러스터 a{i}', index=False)

print("클러스터별 다중회귀 결과가 엑셀 파일로 저장되었습니다.")
