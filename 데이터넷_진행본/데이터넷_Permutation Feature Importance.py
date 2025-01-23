# # 필요한 라이브러리 불러오기
# import os
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor  # RandomForestClassifier 대신 사용
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.inspection import permutation_importance
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler  # 표준화를 위해 추가
#
# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family= 'Malgun Gothic')
#
# # 데이터셋 로드
# dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
# filename = "데이터넷_1_01_시설정보.xlsx"
# file_path = os.path.join(dir_path, filename)
# q1 = pd.read_excel(file_path)
# # data = q1[q1['종별코드명'].isin(['한방병원', '정신병원', '치과병원'])].copy()  # .copy()를 사용해 복사본 생성
# # data = q1[q1['종별코드명'].isin(['요양병원'])].copy()
# # data = q1[q1['종별코드명'].isin(['병원','요양병원'])].copy()
# data = q1[q1['종별코드명'].isin(['종합병원'])].copy()
# # data=q1.copy()
# data = data[data['year_use'] == 2022]
#
# # 새로운 컬럼 생성 (SettingWithCopyWarning 방지)
# data['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
# data['의사수'] = data['총의사수']
# data['병상수'] = data['총병상수']
# data['용적률산정연면적'] = data['용적률산정연면적(㎡)']
# data['대지면적'] = data['대지면적(㎡)']
# data['연면적'] = data['연면적(㎡)']
# data['층수'] = data['지상층수'] + data['지하층수']
#
#
#
#
#
# #
# #
# #
# # # 설명 변수와 목표 변수 설정
# # Q1 = data['USE_QTY_kWh'].quantile(0.25)
# # Q3 = data['USE_QTY_kWh'].quantile(0.75)
# #
# # # IQR 계산
# # IQR = Q3 - Q1
# #
# # # 이상치 경계값 계산
# # lower_bound = Q1 - 1.5 * IQR
# # upper_bound = Q3 + 1.5 * IQR
# #
# # # 이상치 제거
# # data = data[(data['USE_QTY_kWh'] >= lower_bound) & (data['USE_QTY_kWh'] <= upper_bound)]
# #
# #
# #
#
#
# # 특성과 타겟 변수 설정
# # X = data[['승강기수', '의사수', '병상수', "용적률산정연면적", "대지면적","층수"]]
# # X = data[['승강기수', '의사수', '병상수', "연면적", "대지면적","층수"]]
# X = data[['승강기수', '의사수', '병상수', "용적률산정연면적", "대지면적", "연면적", "지하층수", "지상층수", "층수"]]
#
# # X = data[["지하층수", "지상층수", "층수"]]
# # X = data[["용적률산정연면적", "대지면적", "연면적"]]
#
#
#
#
#
#
#
# y = data['USE_QTY_kWh']  # 연속형 타겟 변수 (에너지 사용량)
#
# # 훈련 세트와 테스트 세트로 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # # 데이터 표준화
# # scaler = StandardScaler()
# #
# # # 훈련 세트와 테스트 세트에 대해 별도로 표준화를 수행
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
#
# # 모델 훈련
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)
#
# # 모델의 초기 성능 평가
# y_pred = model.predict(X_test)
# initial_mse = mean_squared_error(y_test, y_pred)
# print(f"Initial Mean Squared Error: {initial_mse:.4f}")
#
# perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
#
# # 결과 출력
# importance_df = pd.DataFrame({'Feature': X.columns,
#                               'Importance Mean': perm_importance.importances_mean,
#                               'Importance Std': perm_importance.importances_std})
#
# importance_df.sort_values(by='Importance Mean', ascending=False, inplace=True)
# print(importance_df)
#
# # 시각화 (옵션)
# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Feature'], importance_df['Importance Mean'], xerr=importance_df['Importance Std'])
# plt.xlabel("Permutation Importance")
# plt.ylabel("Features")
# plt.title("Feature Importance based on Permutation")
# plt.gca().invert_yaxis()
# plt.show()
import os

# # 필요한 라이브러리 불러오기
# import os
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor  # RandomForestClassifier 대신 사용
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
#
# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family= 'Malgun Gothic')
#
# # 데이터셋 로드
# dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
# filename = "데이터넷_1_01_시설정보.xlsx"
# file_path = os.path.join(dir_path, filename)
# q1 = pd.read_excel(file_path)
#
# # 필요한 데이터 필터링
# data = q1.copy()
# data = data[data['year_use'] == 2022]
#
# # 새로운 컬럼 생성 (SettingWithCopyWarning 방지)
# data['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
# data['의사수'] = data['총의사수']
# data['병상수'] = data['총병상수']
# data['용적률산정연면적'] = data['용적률산정연면적(㎡)']
# data['대지면적'] = data['대지면적(㎡)']
# data['연면적'] = data['연면적(㎡)']
# data['층수'] = data['지상층수'] + data['지하층수']
#
#
#
# # 설명 변수와 목표 변수 설정
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
#
#
# # 특성과 타겟 변수 설정
# X = data[['승강기수', '의사수', '병상수', "용적률산정연면적", "대지면적", "연면적", "지하층수", "지상층수", "층수"]]
# y = data['USE_QTY_kWh']  # 연속형 타겟 변수 (에너지 사용량)
#
# # 훈련 세트와 테스트 세트로 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 모델 훈련
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)
#
# # 모델의 초기 성능 평가
# y_pred = model.predict(X_test)
# initial_mse = mean_squared_error(y_test, y_pred)
# print(f"Initial Mean Squared Error: {initial_mse:.4f}")
#
# # Drop Column Importance 계산
# drop_importance = []
#
# for column in X.columns:
#     # 특정 컬럼을 제외한 데이터셋 준비
#     X_train_drop = X_train.drop(columns=[column])
#     X_test_drop = X_test.drop(columns=[column])
#
#     # 모델 훈련 및 예측
#     model_drop = RandomForestRegressor(random_state=42)
#     model_drop.fit(X_train_drop, y_train)
#     y_pred_drop = model_drop.predict(X_test_drop)
#
#     # 모델 성능 평가
#     drop_mse = mean_squared_error(y_test, y_pred_drop)
#
#     # 성능 차이 계산
#     importance = drop_mse - initial_mse
#     drop_importance.append(importance)
#
# # 중요도 데이터프레임 생성
# importance_df = pd.DataFrame({'Feature': X.columns,
#                               'Drop Column Importance': drop_importance})
#
# importance_df.sort_values(by='Drop Column Importance', ascending=False, inplace=True)
# print(importance_df)
#
# # 시각화 (옵션)
# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Feature'], importance_df['Drop Column Importance'])
# plt.xlabel("Drop Column Importance")
# plt.ylabel("Features")
# plt.title("Feature Importance based on Drop Column Method")
# plt.gca().invert_yaxis()
# plt.show()

#
# # 필요한 라이브러리 불러오기
# import os
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family= 'Malgun Gothic')
#
# # 데이터셋 로드
# dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
# filename = "데이터넷_1_01_시설정보.xlsx"
# file_path = os.path.join(dir_path, filename)
# q1 = pd.read_excel(file_path)
#
# # 필요한 데이터 필터링
# data = q1.copy()
# data = data[data['year_use'] == 2022]
#
# # 새로운 컬럼 생성 (SettingWithCopyWarning 방지)
# data['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
# data['의사수'] = data['총의사수']
# data['병상수'] = data['총병상수']
# data['용적률산정연면적'] = data['용적률산정연면적(㎡)']
# data['대지면적'] = data['대지면적(㎡)']
# data['연면적'] = data['연면적(㎡)']
# data['층수'] = data['지상층수'] + data['지하층수']
#
# #
# # # 설명 변수와 목표 변수 설정
# # Q1 = data['USE_QTY_kWh'].quantile(0.25)
# # Q3 = data['USE_QTY_kWh'].quantile(0.75)
# #
# # # IQR 계산
# # IQR = Q3 - Q1
# #
# # # 이상치 경계값 계산
# # lower_bound = Q1 - 1.5 * IQR
# # upper_bound = Q3 + 1.5 * IQR
# #
# # # 이상치 제거
# # data = data[(data['USE_QTY_kWh'] >= lower_bound) & (data['USE_QTY_kWh'] <= upper_bound)]
#
#
# # 특성과 타겟 변수 설정
# X = data[['승강기수', '의사수', '병상수', "용적률산정연면적", "대지면적", "연면적", "지하층수", "지상층수", "층수"]]
# y = data['USE_QTY_kWh']  # 연속형 타겟 변수 (에너지 사용량)
#
# # 훈련 세트와 테스트 세트로 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 모델 훈련
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)
#
# # MDI(Mean Decrease in Impurity) Importance 계산
# mdi_importance = model.feature_importances_
#
# # 중요도 데이터프레임 생성
# importance_df = pd.DataFrame({'Feature': X.columns,
#                               'MDI Importance': mdi_importance})
#
# importance_df.sort_values(by='MDI Importance', ascending=False, inplace=True)
# print(importance_df)
#
# # 시각화 (옵션)
# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Feature'], importance_df['MDI Importance'])
# plt.xlabel("MDI Importance")
# plt.ylabel("Features")
# plt.title("Feature Importance based on MDI (Mean Decrease in Impurity)")
# plt.gca().invert_yaxis()
# plt.show()

########  RFE(Recursive Feature Elimination)  ##########################################################################
#
#
# # 필요한 라이브러리 불러오기
# import os
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.inspection import permutation_importance
# from sklearn.feature_selection import RFE
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family= 'Malgun Gothic')
#
# # 데이터셋 로드 (경로는 사용자 환경에 맞게 수정)
# dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
# filename = "데이터넷_1_01_시설정보.xlsx"
# file_path = os.path.join(dir_path, filename)
# q1 = pd.read_excel(file_path)
#
# # 데이터 필터링 및 전처리
# data = q1[q1['종별코드명'].isin(['종합병원'])].copy()
# data = data[data['year_use'] == 2022]
#
# data['승강기수'] = data['비상용승강기수'] + data['승용승강기수']
# data['의사수'] = data['총의사수']
# data['병상수'] = data['총병상수']
# data['용적률산정연면적'] = data['용적률산정연면적(㎡)']
# data['대지면적'] = data['대지면적(㎡)']
# data['연면적'] = data['연면적(㎡)']
# data['층수'] = data['지상층수'] + data['지하층수']
#
# # 특성과 타겟 변수 설정
# X = data[['승강기수', '의사수', '병상수', "용적률산정연면적", "대지면적", "연면적", "지하층수", "지상층수", "층수"]]
# y = data['USE_QTY_kWh']
#
# # 훈련 세트와 테스트 세트로 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 모델 초기화
# model = RandomForestRegressor(random_state=42)
#
# # Recursive Feature Elimination (RFE) 적용
# selector = RFE(estimator=model, n_features_to_select=1, step=1)  # 한 번에 하나의 변수를 제거하며 평가
# selector = selector.fit(X_train, y_train)
#
# # RFE 결과 출력
# ranking = selector.ranking_
# features_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': ranking})
# features_ranking.sort_values(by='Ranking', ascending=True, inplace=True)
# print(features_ranking)
#
# # 시각화
# plt.figure(figsize=(10, 6))
# plt.barh(features_ranking['Feature'], features_ranking['Ranking'])
# plt.xlabel("Ranking")
# plt.ylabel("Features")
# plt.title("Feature Ranking using RFE")
# plt.gca().invert_yaxis()
# plt.show()



import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

# 데이터셋 로드 (경로는 사용자 환경에 맞게 수정)
dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
q1 = pd.read_excel(file_path)

# 데이터 필터링 및 전처리
# data = q1[q1['종별코드명'].isin(['종합병원'])].copy()
data = q1.copy()
data = data[data['year_use'] == 2022]

# .loc 사용하여 새로운 열 생성
data.loc[:, '승강기수'] = data['비상용승강기수'] + data['승용승강기수']
data.loc[:, '의사수'] = data['총의사수']
data.loc[:, '병상수'] = data['총병상수']
data.loc[:, '용적률산정연면적'] = data['용적률산정연면적(㎡)']
data.loc[:, '대지면적'] = data['대지면적(㎡)']
data.loc[:, '연면적'] = data['연면적(㎡)']
data.loc[:, '층수'] = data['지상층수'] + data['지하층수']

X = data[['승강기수', '의사수', '병상수', "용적률산정연면적", "대지면적", "연면적", "지하층수", "지상층수", "층수"]]
y = data['USE_QTY_kWh']

# 훈련 세트와 테스트 세트로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = RandomForestRegressor(random_state=42)

# RFECV 정의
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='neg_mean_squared_error')
rfecv.fit(X_train, y_train)

# 선택된 최적의 변수 개수
print(f"Optimal number of features: {rfecv.n_features_}")

# 선택된 변수
selected_features = X_train.columns[rfecv.support_]
print(f"Selected features: {selected_features}")

# RFECV 결과 시각화
plt.figure(figsize=(10, 6))
plt.title('RFECV - Optimal Number of Features')
plt.xlabel('Number of features selected')
plt.ylabel('Cross-validation score (neg_mean_squared_error)')

# 'cv_results_'를 사용하여 교차 검증 점수를 가져옵니다.
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()
