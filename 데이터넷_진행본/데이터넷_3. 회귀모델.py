import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')


def OLS_re(year, target):
    # 디렉토리 경로 및 파일명 설정
    dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\{}".format(year)
    filenames_bldg = "데이터넷_2_{}_02_세부정보.xlsx".format(year)
    file_path = os.path.join(dir_path, filenames_bldg)
    df_com = pd.read_excel(file_path)
    if target == '총의사수':
        df_com = df_com[df_com['총의사수'] != 0]
    if target == '건축면적(㎡)':
        df_com = df_com[df_com['건축면적(㎡)'] != 0]
        df_com = df_com[~df_com['건축면적(㎡)'].isna()]  # '에너지공급기관코드'열에 모두 value가 존재하므로 의미가 없음.
    # df_com['면적당성인중환자병상수'] = df_com['성인중환자병상수'] / df_com['연면적(㎡)']
    df_com['면적당의과전문의 인원수'] = df_com['의과전문의 인원수'] / df_com['용적률산정연면적(㎡)']

    df_com['EUI1'] = df_com['USE_QTY_kWh'] / df_com['연면적(㎡)']
    df_com['EUI2'] = df_com['USE_QTY_kWh'] / df_com['용적률산정연면적(㎡)']
    df_com['USE_QTY_kWh(비율)']=df_com['USE_QTY_kWh']*df_com['주용도(의료시설) 비율(%)']

    B = 'USE_QTY_kWh(비율)'
    df_cleaned = df_com.dropna(subset=[target, B])

    # 회귀분석을 위한 X와 y 준비
    X = df_cleaned[target]
    y = df_cleaned[B]
    X = sm.add_constant(X)  # 상수항 추가

    # OLS 모델 피팅
    model = sm.OLS(y, X).fit()

    #
    # # 산점도와 회귀선 플롯
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[target], y, alpha=0.5)
    # plt.title('Scatter Plot')
    # plt.xlabel(target)
    # plt.ylabel('Energy Usage (kWh)')
    #
    # # 회귀선 플롯
    # x_pred = np.linspace(X[target].min(), X[target].max(), 100)
    # y_pred = model.predict(sm.add_constant(x_pred.reshape(-1, 1), has_constant='add'))
    # plt.plot(x_pred, y_pred, 'r', lw=2)  # 빨간색 회귀선
    # plt.tight_layout()
    #
    # # 플롯 저장
    # filenames_bldg = target + "{}.tiff".format(year)
    # file_path = os.path.join(dir_path, filenames_bldg)
    # plt.savefig(file_path, format='tiff', dpi=300)
    #
    # # 로컬에서 플롯을 표시
    # plt.show()
    #
    # # 플롯 닫기
    # plt.close()
    # print(model.rsquared)
    # # R² 값과 Prob(F-statistic) 값 반환/
    return model.rsquared, model.f_pvalue

#
# OLS_re(2018, '면적당의과전문의 인원수')
# OLS_re(2019, '면적당의과전문의 인원수')
# OLS_re(2020, '면적당의과전문의 인원수')
# OLS_re(2021, '면적당의과전문의 인원수')
# OLS_re(2022, '면적당의과전문의 인원수')


A = ['주용도(의료시설) 비율(%)', 'electricity_kWh', 'gas_kWh', 'districtheat_kWh', '대지면적(㎡)', '건축면적(㎡)', '건폐율(%)', '연면적(㎡)', '용적률산정연면적(㎡)', '용적률(%)', '높이(m)', '지상층수', '지하층수', '승용승강기수', '비상용승강기수', '부속건축물수', '부속건축물면적(㎡)', '총동연면적(㎡)']
B = ['총의사수', '의과일반의 인원수', '의과인턴 인원수', '의과레지던트 인원수', '의과전문의 인원수', '치과일반의 인원수', '치과인턴 인원수', '치과레지던트 인원수', '치과전문의 인원수','한방일반의 인원수', '한방인턴 인원수', '한방레지던트 인원수', '한방전문의 인원수']
# C = ['총병상수','일반입원실상급병상수', '일반입원실일반병상수', '성인중환자병상수', '소아중환자병상수', '신생아중환자병상수', '정신과폐쇄상급병상수', '정신과폐쇄일반병상수', '격리병실병상수', '무균치료실병상수', '분만실병상수', '수술실병상수', '응급실병상수', '물리치료실병상수']
C = ['진료시간']



# 수집할 데이터를 저장할 딕셔너리 초기화
data = {target: [] for target in A + B + C}
years = [2018, 2019, 2020, 2021, 2022]

# R² 값 수집
for colums_data in data.keys():
    for year in years:
        r2, p_value = OLS_re(year, colums_data)
        data[colums_data].append(r2)

# 수집된 데이터를 데이터프레임으로 변환
df_r2 = pd.DataFrame(data, index=years).T
df_r2.columns = map(str, years)

# 엑셀 파일로 저장
output_path = r"C:\Users\user\Desktop\R2_values.xlsx"
df_r2.to_excel(output_path, sheet_name='R2_Values')

# 결과 출력 (선택 사항)
print(df_r2)
