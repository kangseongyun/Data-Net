import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')


# 데이터 경로 및 파일명 설정
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)

# 엑셀 파일 읽기
q1 = pd.read_excel(file_path)


def remove_outliers(hos):
    Q1 = hos['USE_QTY_kWh'].quantile(0.25)
    Q3 = hos['USE_QTY_kWh'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    hos = hos[(hos['USE_QTY_kWh'] >= lower_bound) & (hos['USE_QTY_kWh'] <= upper_bound)]
    return hos
grouped_mean = q1['USE_QTY_kWh'].groupby(q1['종별코드명']).mean()
print(grouped_mean)

grouped_median = q1['USE_QTY_kWh'].groupby(q1['종별코드명']).median()
print(grouped_median)
# # 연도 설정
# # 데이터 필터링 및 그룹 추가
# Group_1 = q1[q1['종별코드명'].isin(['종합병원'])].copy()
# Group_1=remove_outliers(Group_1)
# Group_1.loc[:, 'Group'] = 'Group_1'
#
# Group_2 = q1[q1['종별코드명'].isin(['병원','한방병원', '요양병원'])].copy()
# Group_2=remove_outliers(Group_2)
# Group_2.loc[:, 'Group'] = 'Group_2'
#
# Group_3 = q1[q1['종별코드명'].isin(['정신병원', '치과병원'])].copy()
# Group_3=remove_outliers(Group_3)
# Group_3.loc[:, 'Group'] = 'Group_3'
# # 그룹 병합


# 연도 설정
# 데이터 필터링 및 그룹 추가
Group_1 = q1[q1['종별코드명'].isin(['병원','종합병원'])].copy()
# Group_1=remove_outliers(Group_1)
Group_1.loc[:, 'Group'] = 'Group_1'

Group_2 = q1[q1['종별코드명'].isin(['한방병원', '요양병원'])].copy()
# Group_2=remove_outliers(Group_2)
Group_2.loc[:, 'Group'] = 'Group_2'

Group_3 = q1[q1['종별코드명'].isin(['정신병원','치과병원'])].copy()
# Group_3=remove_outliers(Group_3)
Group_3.loc[:, 'Group'] = 'Group_3'
# 그룹 병합



combined_data = pd.concat([Group_1, Group_2, Group_3])


grouped_mean = q1['USE_QTY_kWh'].groupby(q1['종별코드명']).mean()
print(grouped_mean)


# 박스플롯 그리기
plt.figure(figsize=(10, 6))
# sns.boxplot(x="Group", y="USE_QTY_kWh", data=combined_data)
sns.violinplot(x="Group", y="USE_QTY_kWh", data=combined_data)

plt.tight_layout()
plt.show()

#
# # def remove_outliers(hos):
# #     Q1 = hos['USE_QTY_kWh'].quantile(0.25)
# #     Q3 = hos['USE_QTY_kWh'].quantile(0.75)
# #     IQR = Q3 - Q1
# #     lower_bound = Q1 - 1.5 * IQR
# #     upper_bound = Q3 + 1.5 * IQR
# #     hos = hos[(hos['USE_QTY_kWh'] >= lower_bound) & (hos['USE_QTY_kWh'] <= upper_bound)]
# #     return hos
# #
# #
#
# sns.boxplot(x="종별코드명", y="USE_QTY_kWh", data=q1)
#
#
# # sns.boxplot(x="종별코드명", y="USE_QTY_kWh", data=remove_outliers(q1))
# sns.boxplot(x="종별코드명", y="USE_QTY_kWh", data=Group_1)
#
#
# plt.tight_layout()
# plt.show()