import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
import openpyxl
import xlsxwriter
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')

file1 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결\1.병원정보서비스 2022.10..csv", encoding='EUC-KR', dtype=str)
# print(file1)

file2 = pd.read_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\건강보험심사평가원_전국 병의원 및 약국 현황-PK연결\4.의료기관별상세정보서비스_02_세부정보_202309.csv", encoding='EUC-KR', dtype=str)
# print(file2)

merged_df = pd.merge(file1, file2, on=['암호화요양기호', '요양기관명'])
print(merged_df)

file3 = pd.read_excel(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터 편집]에너지사용량_2018.xlsx", dtype=str)
merged_df['mgm_bld_pk'] = merged_df['mgm_bld_pk'].str.split(',')
merged_df1 = merged_df.explode('mgm_bld_pk')
merged_df2 = pd.merge(merged_df1, file3, left_on='mgm_bld_pk', right_on='매칭총괄표제부PK', how='inner')
columns_to_remove = ['mgm_bld_pk', 'mgm_upper_bld_pk']
merged_df2=merged_df2.drop(columns=columns_to_remove)
print(merged_df2)
merged_df2['사용량'] = merged_df2['사용량'].astype(float)
result = merged_df2.groupby(list(merged_df2.columns[0:62]), dropna=False)['사용량'].sum().reset_index()
result.to_csv(r"C:\Users\user\OneDrive - Ajou University\학부연구생 발표\데이터넷 과제\[데이터병합]2.세부정보.csv", encoding='EUC-KR', index=False)
print(result)
result['사용년월'] = result['사용년월'].astype(str)

# '사용년월'에서 연도와 월을 분리
result['연도'] = result['사용년월'].str[:4]  # 처음 4자리가 연도
result['월'] = result['사용년월'].str[4:]   # 나머지가 월

# '월' 열을 정수형으로 변환
result['월'] = result['월'].astype(int)

# 원하는 월을 기준으로 데이터 필터링
# 예: 10월 데이터만 추출
result = result[result['월'] == 10]
elec = result[result['에너지종류'].str.contains('전기', case=False, na=True)]
gas = result[result['에너지종류'].str.contains('도시가스', case=False, na=True)]
heat = result[result['에너지종류'].str.contains('지역난방', case=False, na=True)]


def kmeans_graph(data, n_clusters):
    # 데이터 프레임의 복사본을 생성합니다.
    data_copy = data.copy()

    # K-means 클러스터링을 수행합니다.
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    data_copy['Cluster'] = kmeans.fit_predict(data_copy[['총의사수', '사용량']])

    # 군집화 결과 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='총의사수', y='사용량', hue='Cluster', data=data_copy, palette='viridis')
    plt.title('K-Means Clustering Results')
    plt.show()

    # 클러스터별로 데이터를 분할하고 그래프를 그립니다.
    for cluster in range(n_clusters):
        cluster_data = data_copy[data_copy['Cluster'] == cluster]
        graph(cluster_data, f'Cluster {cluster}')


def graph(A, title):
    x1 = pd.to_numeric(A['총의사수'], errors='coerce')
    y = pd.to_numeric(A['사용량'], errors='coerce')

    x = sm.add_constant(x1)
    model = sm.OLS(y, x).fit()
    print(model.summary())
    predictions = model.predict(x)  # get predictions

    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r_squared = model.rsquared
    print(model.summary())
    print('R-squared:', r_squared)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)

    # 선형 회귀 모델 결과 출력
    print(model.summary())

    # 산점도와 회귀선 그래프 생성
    plt.figure(figsize=(12, 10))
    plt.scatter(x1, y, label='Data')
    plt.plot(x1, predictions, color='red', label='Regression Line')
    plt.xlabel('Total Equipment Count', fontsize=15)
    plt.ylabel('Usage', fontsize=15)
    plt.title(title)
    plt.legend()
    plt.show()

kmeans_graph(elec, n_clusters=2)  # n_clusters는 원하는 클러스터 수
