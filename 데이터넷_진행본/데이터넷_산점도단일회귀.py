# import os
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family='Malgun Gothic')
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
# df_new['승강기수'] = filtered_df['비상용승강기수'] + filtered_df['승용승강기수']
# df_new['의사수'] = filtered_df['총의사수']
# df_new['병상수'] = filtered_df['총병상수']
# df_new['용적률산정연면적'] = filtered_df['용적률산정연면적(㎡)']
# df_new['대지면적'] = filtered_df['대지면적(㎡)']
# df_new['연면적'] = filtered_df['연면적(㎡)']
# df_new['지하층수'] = filtered_df['지하층수']
# df_new['지상층수'] = filtered_df['지상층수']
# df_new['층수'] = df_new['지상층수'] + df_new['지하층수']
# df_new['주용도비율'] = filtered_df['주용도(의료시설) 비율(%)']
# df_new['USE_QTY_kWh'] = filtered_df['USE_QTY_kWh']
#
# independent_vars = ['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수', '주용도비율']
#
# results = []
#
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(50, 15))
# axes = axes.flatten()
#
# for idx, var in enumerate(independent_vars):
#     sns.regplot(x=df_new[var], y=df_new['USE_QTY_kWh'], ax=axes[idx], ci=None)
#     axes[idx].set_title(f'{var} vs USE_QTY_kWh', fontsize=20)  # 제목 글자 크기 설정
#     axes[idx].set_xlabel(var, fontsize=15)  # x축 레이블 글자 크기 설정
#     axes[idx].set_ylabel('USE_QTY_kWh', fontsize=15)  # y축 레이블 글자 크기 설정
#
#     # 회귀 분석
#     X = df_new[[var]]
#     y = df_new['USE_QTY_kWh']
#     model = sm.OLS(y, sm.add_constant(X)).fit()
#
#     # 회귀 계수 및 R^2 값 저장
#     results.append({
#         '변수': var,
#         '회귀 계수': model.params[1],
#         'R^2': model.rsquared
#     })
#
#     # 회귀선과 회귀식, R^2 값 표시
#     intercept = model.params[0]
#     slope = model.params[1]
#     r_squared = model.rsquared
#     eq = f'y = {intercept:.2f} + {slope:.2f}x'
#     r2_text = f'R² = {r_squared:.2f}'
#
#     axes[idx].text(0.05, 0.95, eq, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')
#     axes[idx].text(0.05, 0.85, r2_text, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')
#
# plt.tight_layout()
# plt.show()
#
# # 결과를 데이터프레임으로 변환
# results_df = pd.DataFrame(results)
#
# # 엑셀 파일로 저장
# output_filename = "회귀분석결과.xlsx"
# output_filepath = os.path.join(dir_path, output_filename)
# results_df.T.to_excel(output_filepath, index=True)
#
# print(f"회귀분석 결과가 {output_filepath}에 저장되었습니다.")



import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')

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
df_new['연면적'] = filtered_df['연면적(㎡)']
df_new['용적률산정연면적'] = filtered_df['용적률산정연면적(㎡)']
df_new['대지면적'] = filtered_df['대지면적(㎡)']
df_new['층수'] = filtered_df['지상층수'] + filtered_df['지하층수']
df_new['지하층수'] = filtered_df['지하층수']
df_new['지상층수'] = filtered_df['지상층수']
df_new['승강기수'] = filtered_df['비상용승강기수'] + filtered_df['승용승강기수']
df_new['의사수'] = filtered_df['총의사수']
df_new['병상수'] = filtered_df['총병상수']


# df_new['주용도비율'] = filtered_df['주용도(의료시설) 비율(%)']
df_new['USE_QTY_kWh'] = filtered_df['USE_QTY_kWh']


print(df_new['USE_QTY_kWh'].describe())
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
# filtered_df = df_new[(df_new['USE_QTY_kWh'] >= lower_bound) & (df_new['USE_QTY_kWh'] <= upper_bound)]
#


correlation_matrix = df_new.corr()
correlation_matrix.to_excel('correlation_matrix.xlsx')


# 종속 변수 로그 변환
# df_new['log_USE_QTY_kWh'] = np.log(df_new['USE_QTY_kWh'])

independent_vars = ['연면적', '용적률산정연면적', '대지면적', '층수','지하층수', '지상층수', '승강기수', '의사수', '병상수']

results = []

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 25))
axes = axes.flatten()

for idx, var in enumerate(independent_vars):
    # sns.regplot(x=df_new[var], y=df_new['log_USE_QTY_kWh'], ax=axes[idx], ci=None)
    sns.regplot(x=df_new[var], y=df_new['USE_QTY_kWh'], ax=axes[idx], ci=None)
    axes[idx].set_title(f'{var} vs USE_QTY_kWh', fontsize=20)  # 제목 글자 크기 설정
    axes[idx].set_xlabel(var, fontsize=15)  # x축 레이블 글자 크기 설정
    axes[idx].set_ylabel('USE_QTY_kWh', fontsize=15)  # y축 레이블 글자 크기 설정

    # 회귀 분석
    X = df_new[[var]]
    y = df_new['USE_QTY_kWh']
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # 회귀 계수 및 R^2 값 저장
    results.append({
        '변수': var,
        '회귀 계수': model.params[1],
        'R^2': model.rsquared
    })

    # 회귀선과 회귀식, R^2 값 표시
    intercept = model.params[0]
    slope = model.params[1]
    r_squared = model.rsquared
    eq = f'y = {intercept:.2f} + {slope:.2f}x'
    r2_text = f'R² = {r_squared:.2f}'

    axes[idx].text(0.05, 0.95, eq, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')
    axes[idx].text(0.05, 0.85, r2_text, transform=axes[idx].transAxes, fontsize=30, verticalalignment='top')

plt.tight_layout()
plt.show()

# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame(results)
#
# # 엑셀 파일로 저장
# output_filename = "회귀분석결과_log_USE_QTY_kWh.xlsx"
# output_filepath = os.path.join(dir_path, output_filename)
# results_df.T.to_excel(output_filepath, index=True)
#
# print(f"회귀분석 결과가 {output_filepath}에 저장되었습니다.")
