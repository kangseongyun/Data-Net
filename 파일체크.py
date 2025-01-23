import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')




dir_path =r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"
filename = "데이터넷_최종_결합본.xlsx"
file_path = os.path.join(dir_path,filename)

hos_00 = pd.read_excel(file_path)
print("# of PK0 : ", hos_00['매칭표제부PK'].nunique())
print('# of data0 : ', hos_00.shape[0])
print(' ')



filenames_bldg="데이터넷_의료시설(건축물대장).xlsx"

file_path = os.path.join(dir_path, filenames_bldg)
df_bldg1 = pd.read_excel(file_path, sheet_name='층별개요', dtype=str)

print("# of PK1 : ", df_bldg1['매칭표제부PK'].nunique())
print('# of data1 : ', df_bldg1.shape[0])
print(' ')


df_bldg1['면적(㎡)']=df_bldg1['면적(㎡)'].astype(float)
df_bldg1 = df_bldg1[df_bldg1['면적(㎡)'] > 0]

print("# of PK2 : ", df_bldg1['매칭표제부PK'].nunique()) # of PK2 :  7581
print('# of data2 : ', df_bldg1.shape[0]) # of data2 :  57007
print(' ')


PKlist=hos_00['매칭표제부PK'].unique()
floor=df_bldg1[df_bldg1['매칭표제부PK'].isin(PKlist)]
print(floor)
print("# of PK3 : ", floor['매칭표제부PK'].nunique())
print('# of data3 : ', floor.shape[0])
print(' ')


floor1=floor['주용도코드명'].unique()
print(floor1.tolist())


floor=floor[~floor['주용도코드명'].isna()]
print("# of PK4 : ", floor['주용도코드명'].nunique())
print('# of data4 : ', floor.shape[0])
print(' ')



floor2 = floor['주용도코드명'].value_counts()
# floor3 = floor2.reset_index()
# floor3=floor3.T
# floor3.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\floor1.xlsx")

top_categories = floor2[:-85]  # All except the bottom 20
bottom_categories = floor2[-85:]  # The bottom 20 categories

# Summing up the bottom 20 categories into a single category 'Others'
others_count = bottom_categories.sum()

# Creating a new Series with 'Others'
new_value_counts = pd.concat([top_categories, pd.Series({'Others': others_count})])
# new_value_counts1=new_value_counts.reset_index().T
# new_value_counts1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\floor.xlsx")


plt.figure(figsize=(10, 6))
new_value_counts.plot(kind='bar', color='skyblue')  # You can choose any color
plt.title('Count of 주용도코드명')
plt.xlabel('주용도코드명')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




floor_group=floor.pivot_table(index='매칭표제부PK', columns='주용도코드명',values='면적(㎡)',aggfunc='sum')

hos_list=['병원','요양병원','종합병원','기타병원','의료시설','치과병원','한방병원','정신병원','산부인과병원']

hospital_related = floor_group[hos_list].sum(axis=1)

# 병원 관련 열들을 제외한 나머지 열을 합침
other = floor_group.drop(columns=hos_list).sum(axis=1)

# 결과를 새로운 데이터프레임으로 만듦
result = pd.DataFrame({
    '병원 관련 면적': hospital_related,
    '기타 면적': other
}, index=floor_group.index)

result['면적 합계']=result.sum(axis=1)


result_ratio=pd.DataFrame()
result_ratio['병원 면적 비율']=result['병원 관련 면적']/result['면적 합계']
result_ratio['기타 면적 비율']=result['기타 면적']/result['면적 합계']
result_ratio1=result_ratio
# result_ratio1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\floor.xlsx")



plt.figure(figsize=(10, 6))

result_ratio['병원 면적 비율']=result_ratio['병원 면적 비율']*100
result_ratio['기타 면적 비율']=result_ratio['기타 면적 비율']*100

# result_ratio = result_ratio.sort_values(by='병원 면적 비율', ascending=True)
# result_ratio['누적 병원 면적 비율'] = result_ratio['병원 면적 비율'].cumsum()
# plt.scatter(result_ratio['병원 면적 비율'], result_ratio['누적 병원 면적 비율'], color='b', label='병원 면적 비율')
# plt.title('병원 면적 비율 누적 점 그래프')
# plt.xlabel('병원 면적 비율 (%)')
# plt.ylabel('누적 면적 비율 (%)')

result_ratio = result_ratio.sort_values(by='기타 면적 비율', ascending=True)
result_ratio['누적 기타 면적 비율'] = result_ratio['기타 면적 비율'].cumsum()
plt.scatter(result_ratio['기타 면적 비율'], result_ratio['누적 기타 면적 비율'], color='r', label='기타 면적 비율')
plt.title('기타 면적 비율 누적 점 그래프')
plt.xlabel('기타 면적 비율 (%)')
plt.ylabel('누적 면적 비율 (%)')

plt.legend()
plt.grid(True)
plt.show()



# sns.histplot(result_ratio1['병원 면적 비율'], color="blue", label='병원', kde=True, stat="density", binwidth=0.1)
# sns.histplot(result_ratio1['기타 면적 비율'], color="orange", label='기타', kde=True, stat="density", binwidth=0.1)
# plt.xlim([result_ratio1[['병원 면적 비율', '기타 면적 비율']].min().min(), result_ratio1[['병원 면적 비율', '기타 면적 비율']].max().max()])
#
#
# # Additional styling (Optional)
# plt.title('병원시설 주용도 혼합도 히스토그램')  # Title of the histogram in Korean
# plt.xlabel('주용도 혼합도')  # X-axis label in Korean
# plt.ylabel('Density')  # Y-axis label
# plt.legend()  # Show legend
#
# # Show plot
# plt.show()