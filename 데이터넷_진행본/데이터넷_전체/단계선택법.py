import os

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
# data = data[data['종별코드명'].isin(['종합병원'])]
# data = data[data['종별코드명'].isin(['병원'])]
# data = data[data['종별코드명'] == '병원']
# data = data[data['종별코드명'].isin(['요양병원'])]
# data = data[data['종별코드명'].isin(['한방병원'])]
# data = data[data['종별코드명'].isin(['정신병원'])]
# data = data[data['종별코드명'].isin(['치과병원'])]
# data = data[(data['종별코드명'] != '정신병원') & (data['종별코드명'] != '치과병원')]

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

data_n = data_n[(data_n['대지면적'] > 0) & (data_n['의사수'] > 0) & (data_n['지상층수'] > 0)]
print('# of data1 : ', len(data_n))
data = data_n.apply(pd.to_numeric, errors='coerce')

# Define the dependent and independent variables
x = data.drop("USE_QTY_kWh", axis=1)
y = data["USE_QTY_kWh"]

# Perform stepwise regression
result = sm.OLS(y, x).fit()

# Print the summary of the model
print(result.summary())