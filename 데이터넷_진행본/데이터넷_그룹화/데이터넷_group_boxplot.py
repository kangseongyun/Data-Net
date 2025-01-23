import os

import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')
# 데이터 로드
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
data = pd.read_excel(file_path)

data = data[data['year_use'] == 2022]

Group_1 = data[data['종별코드명'].isin(['종합병원'])]

Group_2 = data[data['종별코드명'].isin(['병원','요양병원'])]

Group_3 = data[data['종별코드명'].isin(["한방병원",'정신병원','치과병원'])]





data=data
method='기본'
# method='IQR'


# 필요한 변수 생성
data_n = pd.DataFrame()

data_n['종별코드명']=data['종별코드명']
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
data_n=data_n[['종별코드명', 'USE_QTY_kWh']]
print('# of data1 : ', len(data_n))
data_n = data_n[data_n['종별코드명'] != '종합병원']

# 각 종별코드명 별로 중앙값 계산
median_values = data_n.groupby('종별코드명')['USE_QTY_kWh'].median().sort_values(ascending=False)

# 중앙값 순서대로 정렬
data_n['종별코드명'] = pd.Categorical(data_n['종별코드명'], categories=median_values.index, ordered=True)

# 데이터 개수 계산 및 새로운 레이블 생성
count_by_category = data_n['종별코드명'].value_counts()
data_n['종별코드명_with_count'] = data_n['종별코드명'].map(lambda x: f"{x} ({count_by_category[x]})")

# Stripplot 그리기 (중앙값 순서대로 정렬된 상태)
plt.figure(figsize=(12, 8))
ax = sns.stripplot(x='종별코드명_with_count', y='USE_QTY_kWh', data=data_n, jitter=True,
                   order=data_n['종별코드명_with_count'].cat.categories)

# 중앙값과 평균값 선을 그리기
for category in median_values.index:
    category_data = data_n[data_n['종별코드명'] == category]['USE_QTY_kWh']
    median_value = category_data.median()
    mean_value = category_data.mean()

    # 해당 카테고리의 x 위치 찾기
    xpos = np.where(median_values.index == category)[0][0]

    # 중앙값과 평균값 선 그리기 (점들보다 앞에 나오도록)
    ax.plot([xpos - 0.2, xpos + 0.2], [median_value, median_value], color='blue', linestyle='--', lw=2, zorder=10,
            label=f'Median' if xpos == 0 else "")
    ax.plot([xpos - 0.2, xpos + 0.2], [mean_value, mean_value], color='red', linestyle='-', lw=2, zorder=10,
            label=f'Mean' if xpos == 0 else "")

# 범례 추가
plt.legend()

# xtick 설정
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
