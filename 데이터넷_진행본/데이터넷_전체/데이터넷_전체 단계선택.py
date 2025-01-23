import os

from scipy.stats import boxcox  # scipy.special이 아닌 scipy.stats에서 boxcox를 가져옵니다

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
data = data[data['종별코드명'].isin(['정신병원'])]
# data = data[data['종별코드명'].isin(['치과병원'])]
# data = data[(data['종별코드명'] != '치과병원')]


# data = data[(data['종별코드명'] != '정신병원') & (data['종별코드명'] != '치과병원')]

method='기본'
# method='IQR'


# 필요한 변수 생성
data_n = pd.DataFrame()
data_n['연면적'] = data['연면적(㎡)']
data_n['용적률산정연면적'] = data['용적률산정연면적(㎡)']
data_n['대지면적'] = data['대지면적(㎡)']
data_n['건축면적'] = data['건축면적(㎡)']
data_n['건폐율'] = data['건폐율(%)']
data_n['용적률'] = data['용적률(%)']
data_n['층수'] = data['지상층수']+data['지하층수']
data_n['지하층수'] = data['지하층수']
data_n['지상층수'] = data['지상층수']
data_n['높이'] = data['높이(m)']
data_n['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
data_n['의사수'] = data['총의사수']
data_n['병상수'] = data['총병상수']
data_n['USE_QTY_kWh'] = data['USE_QTY_kWh']

data_n = data_n.dropna(subset=['승강기수', '의사수','건폐율', '용적률', '건축면적', '높이', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
print('# of data1 : ', len(data_n))
# 높이
data_n = data_n[(data_n['대지면적'] > 0) & (data_n['의사수'] > 0) & (data_n['지상층수'] > 0)
                & (data_n['건폐율'] > 0)& (data_n['건축면적'] > 0)& (data_n['용적률'] > 0)]
print('# of data1 : ', len(data_n))
# data_n.to_excel(r"C:\Users\user\Desktop\spss분석용\Total.xlsx")
# data_n.to_excel(r"C:\Users\user\Desktop\spss분석용\종합병원.xlsx")
# data_n.to_excel(r"C:\Users\user\Desktop\spss분석용\병원.xlsx")
# data_n.to_excel(r"C:\Users\user\Desktop\spss분석용\요양병원.xlsx")
# data_n.to_excel(r"C:\Users\user\Desktop\spss분석용\한방병원.xlsx")
# data_n.to_excel(r"C:\Users\user\Desktop\spss분석용\정신병원.xlsx")

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

    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)

    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)

    # Box-Cox transformation of dependent variable
    # hos['USE_QTY_kWh'], lambda_ = boxcox(hos['USE_QTY_kWh'] + 1e-8)  # Add a tiny constant to avoid zero values
    # print(lambda_)
    return hos

data_n=transform_hos(data_n, method)


print(data_n)


def centering_data(a):
    c_data = pd.DataFrame()
    c_data['승강기수'] = a['승강기수']-a['승강기수'].mean()
    print(data_n['승강기수'].mean())
    c_data['의사수'] = a['의사수']-a['의사수'].mean()
    c_data['병상수'] = a['병상수']-a['병상수'].mean()
    c_data['연면적'] = a['연면적']-a['연면적'].mean()
    c_data['용적률산정연면적'] = a['용적률산정연면적']-a['용적률산정연면적'].mean()
    c_data['대지면적'] = a['대지면적']-a['대지면적'].mean()
    c_data['지하층수'] = a['지하층수']-a['지하층수'].mean()
    c_data['지상층수'] = a['지상층수']-a['지상층수'].mean()
    c_data['층수'] = a['층수']-a['층수'].mean()
    c_data['USE_QTY_kWh'] = a['USE_QTY_kWh']
    return c_data


c_data=centering_data(data_n)
print(c_data)









print('# of data1 : ', len(data_n))





# 설명 변수와 목표 변수 설정
X = c_data[['승강기수', '의사수', '병상수','연면적','층수']]


y = c_data['USE_QTY_kWh']

# 단계적 선택법 함수 정의
def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    adj_r_squared_list = []
    ALC_list = []
    step_list = []
    final_model = None

    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add  {best_feature:30} with p-value {best_pval:.6}')
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        adj_r_squared_list.append(model.rsquared_adj)
        ALC_list.append(model.aic)
        step_list.append("\n".join(included))
        final_model = model  # Keep the last model as final model

        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Remove {worst_feature:30} with p-value {worst_pval:.6}')

        if not changed:
            break

    print(final_model.summary())
    return included, adj_r_squared_list, ALC_list, step_list, final_model

# Perform stepwise selection and record Adjusted R-squared values
elected_features, adj_r_squared_list, ALC_list, step_list, final_model = stepwise_selection(X, y)

# Print selected features and their standard errors
print("Selected Features:", elected_features)
print("\nStandard Errors of Selected Features:\n")
print(final_model.bse)

# 동일한 단계 감지 및 해결
for i in range(1, len(step_list)):
    if step_list[i] == step_list[i - 1]:
        step_list[i] += f"\n(중복 {i+1})"  # 동일 단계를 구분하기 위해 고유 식별자 추가

# 조정된 R 제곱 값을 이용해 그래프 그리기
fig, ax1 = plt.subplots(figsize=(15, 10))

# ax1에 대한 그래프
line1, = ax1.plot(range(1, len(adj_r_squared_list) + 1), adj_r_squared_list, marker='o', label='Adjusted R Squared', linewidth=5, color='blue', markersize=20)
ax1.set_ylabel('Adjusted R Squared', fontsize=20)
ax1.tick_params(axis='y', labelsize=20)

# ax2에 대한 그래프
ax2 = ax1.twinx()
line2, = ax2.plot(range(1, len(ALC_list) + 1), ALC_list, marker='^', label='ALC', linewidth=5, color='red', markersize=20)
ax2.set_ylabel('ALC', fontsize=20)
ax2.tick_params(axis='y', labelsize=20)

# x축 및 그래프 제목 설정
plt.xticks(range(1, len(step_list) + 1), step_list, rotation=0, ha="center", fontsize=20)
ax1.tick_params(axis='x', labelsize=20)
plt.xlabel('Step', fontsize=20)

# 두 개의 범례를 하나로 합치기
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right', fontsize=20)

# 그리드, 타이틀 및 레이아웃 설정
plt.grid(True)
plt.tight_layout()

# 그래프 출력
plt.show()
# import pandas as pd
# import statsmodels.api as sm
#
# # 데이터프레임에서 설명 변수와 목표 변수 설정
# # 설명 변수와 목표 변수 설정
# X = data[['승강기수', '의사수', '병상수', "용적률산정연면적", "대지면적", "층수"]]
# # X = data[['병상수','승강기수', '의사수',  "연면적", "대지면적", "층수"]]
# y = data['USE_QTY_kWh']  # 연속형 타겟 변수 (에너지 사용량)
#
#
# # 후진제거법 함수 정의
# def backward_elimination(X, y, threshold_out=0.05, verbose=True):
#     """
#     후진제거법을 이용한 변수 선택
#
#     Args:
#         X - 설명 변수 DataFrame
#         y - 목표 변수 Series
#         threshold_out - 변수 제거 임계값 (기본값: 0.05)
#         verbose - 중간 과정 출력 여부 (기본값: True)
#
#     Returns:
#         포함된 변수 리스트
#     """
#     included = list(X.columns)
#
#     while True:
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
#         pvalues = model.pvalues.iloc[1:]  # 첫 번째는 상수항이므로 제외
#         worst_pval = pvalues.max()
#         if worst_pval > threshold_out:
#             worst_feature = pvalues.idxmax()
#             included.remove(worst_feature)
#             if verbose:
#                 print(f'Remove {worst_feature:30} with p-value {worst_pval:.6}')
#         else:
#             break
#
#     return included
#
#
# # 후진제거법 실행
# selected_features = backward_elimination(X, y)
#
# print(f'최종 선택된 변수: {selected_features}')
#
# import pandas as pd
# import statsmodels.api as sm
# from scipy.stats import boxcox  # scipy.special이 아닌 scipy.stats에서 boxcox를 가져옵니다
#
# # 데이터프레임에서 설명 변수와 목표 변수 설정
# # X = data[['병상수','승강기수', '의사수',  "연면적", "대지면적", "층수"]]
# # data['USE_QTY_kWh'] = data['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)
# # data['USE_QTY_kWh'], lambda_ = boxcox(data['USE_QTY_kWh'])
# Q1 = data['USE_QTY_kWh'].quantile(0.25)
# Q3 = data['USE_QTY_kWh'].quantile(0.75)
#
# # IQR 계산
# IQR = Q3 - Q1
#
# # 이상치 경계값 계산
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # 이상치 제거
# data = data[(data['USE_QTY_kWh'] >= lower_bound) & (data['USE_QTY_kWh'] <= upper_bound)]
# data['USE_QTY_kWh'] = data['USE_QTY_kWh'].apply(lambda x: np.sqrt(x) if x > 0 else 0)
#
# # print(lambda_)
# # X = data[['승강기수','의사수', '병상수', "용적률산정연면적", "대지면적"]]
# X = data[['승강기수','층수','의사수', '병상수', "용적률산정연면적", "대지면적"]]
# # X = data[['승강기수', '의사수', '병상수', "연면적", "대지면적"]]
# # X = data[['승강기수', "층수", '의사수', '병상수', "연면적", "대지면적"]]
#
# y = data['USE_QTY_kWh']  # 연속형 타겟 변수 (에너지 사용량)
#
#
#
#
# # 전진선택법 함수 정의
# def forward_selection(X, y, threshold_in=0.05, verbose=True):
#     """
#     전진선택법을 이용한 변수 선택
#
#     Args:
#         X - 설명 변수 DataFrame
#         y - 목표 변수 Series
#         threshold_in - 변수 포함 임계값 (기본값: 0.05)
#         verbose - 중간 과정 출력 여부 (기본값: True)
#
#     Returns:
#         포함된 변수 리스트
#     """
#     included = []
#     while True:
#         changed = False
#         excluded = list(set(X.columns) - set(included))
#         new_pval = pd.Series(index=excluded, dtype=float)
#         for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
#             new_pval[new_column] = model.pvalues[new_column]
#
#         best_pval = new_pval.min()
#         if best_pval < threshold_in:
#             best_feature = new_pval.idxmin()
#             included.append(best_feature)
#             changed = True
#             if verbose:
#                 print(f'Add  {best_feature:30} with p-value {best_pval:.6}')
#
#         if not changed:
#             break
#
#     return included
#
#
# # 전진선택법 실행
# selected_features = forward_selection(X, y)
# # 선택된 변수에 대한 회귀 모델 피팅
# X_selected = X[selected_features]
# X_selected = sm.add_constant(X_selected)  # 상수항 추가
# model = sm.OLS(y, X_selected).fit()
#
# # 회귀 분석 결과 출력
# print(model.summary())
#
# print(f'최종 선택된 변수: {selected_features}')
