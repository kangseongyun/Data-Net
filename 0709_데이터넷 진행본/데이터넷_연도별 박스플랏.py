import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler

rcParams['font.family'] = 'Malgun Gothic'  # Change to the font name you installed
rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly

dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
df_merge_result = pd.read_excel(file_path)
df_2018 = df_merge_result[df_merge_result['year_use']==2018]
df_2019 = df_merge_result[df_merge_result['year_use']==2019]
df_2020 = df_merge_result[df_merge_result['year_use']==2020]
df_2021 = df_merge_result[df_merge_result['year_use']==2021]
df_2022 = df_merge_result[df_merge_result['year_use']==2022]








def normalize_column(df, column):
    scaler = MinMaxScaler()
    df = df.copy()  # 데이터 프레임 복사본 생성

    df.loc[:, [column]] = scaler.fit_transform(df[[column]])
    return df

# 정규화를 적용할 열 이름 정의
column_to_normalize = 'USE_QTY_kWh'  # 정규화를 적용할 열 이름

# 각 연도별 데이터 프레임에 정규화 적용
df_2018 = normalize_column(df_2018, column_to_normalize)
df_2019 = normalize_column(df_2019, column_to_normalize)
df_2020 = normalize_column(df_2020, column_to_normalize)
df_2021 = normalize_column(df_2021, column_to_normalize)
df_2022 = normalize_column(df_2022, column_to_normalize)



df_clean = pd.concat([df_2018, df_2019, df_2020, df_2021, df_2022])

# 바이올린 그래프 그리기
plt.figure(figsize=(10, 8))
labels=np.arange(2018,2023)

sns.violinplot(x='year_use', y='USE_QTY_kWh', data=df_clean)
# sns.boxplot(x='year_use', y='USE_QTY_kWh', data=df_clean,boxprops={'facecolor':'None'})


# sns.boxplot(x='year_use', y='USE_QTY_kWh', data=df_clean,  showfliers=False)


plt.xlabel('연도')
plt.ylabel('USE_QTY_kWh')
plt.title('연도별 USE_QTY_kWh 분포')

plt.tight_layout()
plt.show()