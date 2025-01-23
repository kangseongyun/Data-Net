import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score


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

# 새로운 데이터프레임 생성
df_new = pd.DataFrame()
df_new['승강기수'] = filtered_df['비상용승강기수'] + filtered_df['승용승강기수']  # all
df_new['의사수'] = filtered_df['총의사수']  # all
df_new['병상수'] = filtered_df['총병상수']  # all
df_new['용적률산정연면적'] = filtered_df['용적률산정연면적(㎡)']
df_new['대지면적'] = filtered_df['대지면적(㎡)']  # all
df_new['연면적'] = filtered_df['연면적(㎡)']  # all
df_new['지하층수'] = filtered_df['지하층수']  # all
df_new['지상층수'] = filtered_df['지상층수']  # all
df_new['층수'] = df_new['지상층수'] + df_new['지하층수']
df_new['USE_QTY_kWh'] = filtered_df['USE_QTY_kWh']

print('검토')
print('# of data : ', df_new.shape[0])
print(' ')

# 결측치 제거
df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수', 'USE_QTY_kWh'])

# 특징과 타겟 변수 분리
X = df_new.drop('USE_QTY_kWh', axis=1)
y = df_new['USE_QTY_kWh']

# 표준화
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 릿지 회귀 모델 학습
ridge = Ridge(max_iter=10000)  # 반복 횟수 증가
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge_grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train, y_train)

# 라쏘 회귀 모델 학습
lasso = Lasso(max_iter=10000)  # 반복 횟수 증가
lasso_grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train, y_train)

# 최적의 모델로 예측
ridge_best = ridge_grid_search.best_estimator_
lasso_best = lasso_grid_search.best_estimator_

y_pred_ridge = ridge_best.predict(X_test)
y_pred_lasso = lasso_best.predict(X_test)

# 평가 지표 계산
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Ridge Regression")
print(f'Best alpha: {ridge_grid_search.best_params_["alpha"]}')
print(f'MSE: {mse_ridge}')
print(f'R²: {r2_ridge}')
print(f'Coefficients: {ridge_best.coef_}')
print("\nLasso Regression")
print(f'Best alpha: {lasso_grid_search.best_params_["alpha"]}')
print(f'MSE: {mse_lasso}')
print(f'R²: {r2_lasso}')
print(f'Coefficients: {lasso_best.coef_}')
