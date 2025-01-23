import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import skew, kurtosis

from scipy.stats import boxcox  # scipy.special이 아닌 scipy.stats에서 boxcox를 가져옵니다
from scipy.stats import pearsonr, shapiro, kstest, anderson, jarque_bera
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')


dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
q1 = pd.read_excel(file_path)
q1 = q1[q1['year_use'].isin([2022])]




# 1사분위수(Q1)와 3사분위수(Q3) 계
def IQR_fitter(q1):
    Q1 = q1['USE_QTY_kWh'].quantile(0.25)
    Q3 = q1['USE_QTY_kWh'].quantile(0.75)

    # IQR 계산
    IQR = Q3 - Q1

    # 이상치 경계값 계산
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치 제거
    q1 = q1[(q1['USE_QTY_kWh'] >= lower_bound) & (q1['USE_QTY_kWh'] <= upper_bound)]



    return q1




def transform_hos(hos):
    hos = hos.copy()

    # Box-Cox transformation of dependent variable
    hos['USE_QTY_kWh'], lambda_ = boxcox(hos['USE_QTY_kWh'])
    print(lambda_)
    # hos['USE_QTY_kWh'] = hos['USE_QTY_kWh'].apply(lambda x: np.log(x) if x > 0 else 0)
    return hos

def Factor(q1):

    df_new = pd.DataFrame()
    df_new['종별코드명'] = q1['종별코드명']
    df_new['연면적'] = q1['연면적(㎡)']
    df_new['용적률산정연면적'] = q1['용적률산정연면적(㎡)']
    df_new['대지면적'] = q1['대지면적(㎡)']
    df_new['층수'] = q1['지상층수'] + q1['지하층수']
    df_new['지하층수'] = q1['지하층수']
    df_new['지상층수'] = q1['지상층수']
    df_new['승강기수'] = q1['비상용승강기수'] + q1['승용승강기수']
    df_new['의사수'] = q1['총의사수']
    df_new['병상수'] = q1['총병상수']
    df_new['USE_QTY_kWh'] = q1['USE_QTY_kWh']
    df_new = df_new.dropna(subset=['승강기수', '의사수', '병상수', '용적률산정연면적', '대지면적', '연면적', '지하층수', '지상층수', '층수'])
    print('# of data1 : ', len(df_new))

    df_new = df_new[(df_new['대지면적'] > 0) & (df_new['의사수'] > 0) & (df_new['지상층수'] > 0)]
    print('# of data1 : ', len(df_new))
    return df_new


# def Factor2_grouped_with_black_kde(A):
#     q1 = Factor(A)
#
#     # 병원 그룹 정의
#     hospital_groups = {
#         'Group 1: 종합병원': ['종합병원'],
#         'Group 2: 병원, 요양병원': ['병원', '요양병원'],
#         'Group 3: 한방병원, 정신병원, 치과병원': ['한방병원', '정신병원', '치과병원']
#     }
#
#     individual_colors = ['cyan', 'red', 'yellow', 'gray', 'purple', 'green']  # 개별 병원 색상
#
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # 1행 3열의 서브플롯 생성
#     axes = axes.flatten()  # 2D 배열을 1D로 변환하여 쉽게 반복문으로 접근 가능
#
#     for i, (group_name, hospitals) in enumerate(hospital_groups.items()):
#         for j, hospital in enumerate(hospitals):
#             individual_data = q1[q1['종별코드명'] == hospital]
#             count = len(individual_data)
#             label = f"{hospital} (n={count})"
#             sns.histplot(individual_data['USE_QTY_kWh'], kde=True, color=individual_colors[j], alpha=0.8, bins=50,
#                          label=label, ax=axes[i])
#
#             # KDE 라인의 색상을 검정으로 변경
#             for line in axes[i].lines:
#                 line.set_color('black')
#
#         axes[i].set_title(group_name)
#         axes[i].set_xlabel('USE_QTY_kWh')
#         axes[i].set_ylabel('Frequency')
#         axes[i].legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# Factor2_grouped_with_black_kde(q1)


# ###### 개별 그래프 출력
# def Factor2(A,B,C):
#     q1=Factor(A)
#     Group_1 = q1[q1['종별코드명'].isin([B])]
#     ax = sns.histplot(Group_1['USE_QTY_kWh'], kde= True, color = C, alpha = 0.8, bins=50, label=B)
#     ax.lines[0].set_color('Black')
#     ax.lines[0].set_linewidth(3)
#     plt.xlabel('USE_QTY_kWh')
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.show()
#
# Factor2( q1, '종합병원', 'cyan')
# Factor2( q1, '치과병원', 'red')
# Factor2( q1, '한방병원', 'yellow')
# Factor2( q1, '정신병원', 'gray')
# Factor2( q1, '한방병원', 'black')
# Factor2( q1, '병원', 'blue')



def Factor2_combined(A):
    q1 = Factor(A)
    hospital_types = ['종합병원', '병원', '요양병원', '한방병원', '정신병원', '치과병원']
    colors = ['cyan', 'red', 'yellow', 'gray', 'blue', 'purple']

    plt.figure(figsize=(10, 6))  # 그래프 크기 조정

    for hospital, color in zip(hospital_types, colors):
        Group_1 = q1[q1['종별코드명'].isin([hospital])]
        sns.histplot(Group_1['USE_QTY_kWh'], color=color, alpha=0.5, bins=50, label=hospital)

    plt.xlabel('USE_QTY_kWh')
    plt.ylabel('Frequency')
    plt.legend(title='병원 종류')
    plt.tight_layout()
    plt.show()

Factor2_combined(q1)




def Factor2_separate(A):
    q1 = Factor(A)
    hospital_types_kor = ['종합병원', '병원', '요양병원', '한방병원', '정신병원', '치과병원']  # 한글 명칭
    hospital_types_eng = ['General Hospital', 'Hospital', 'Long-term Care Hospital', 'Oriental Medicine Hospital', 'Psychiatric Hospital', 'Dental Hospital']  # 영어 명칭
    colors = ['cyan', 'red', 'yellow', 'gray', 'blue', 'purple']
    alphabet_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 알파벳 라벨

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))  # 2행 3열의 서브플롯 생성
    axes = axes.flatten()  # 2D 배열을 1D로 변환하여 쉽게 반복문으로 접근 가능

    for i, (hospital_kor, hospital_eng, color) in enumerate(zip(hospital_types_kor, hospital_types_eng, colors)):
        Group_1 = q1[q1['종별코드명'].isin([hospital_kor])]
        count = len(Group_1)  # 데이터 개수 계산
        sns.histplot(Group_1['USE_QTY_kWh'], kde=True, color=color, alpha=0.8, bins=50, label=hospital_eng, ax=axes[i])

        # KDE 라인의 색상을 검정으로 변경
        for line in axes[i].lines:
            line.set_color('black')

        # 제목에 영어 명칭과 데이터 개수를 포함
        axes[i].set_title(f"{alphabet_labels[i]} {hospital_eng} (n={count})", fontsize=16,fontweight='bold')
        axes[i].set_xlabel('Total annual energy (kWh/yr)', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].tick_params(axis='x', labelsize=12)  # xticks 글자 크기 조정
        axes[i].tick_params(axis='y', labelsize=12)  # yticks 글자 크기 조정
    plt.tight_layout()
    plt.show()

Factor2_separate(q1)




def Factor2_separate(A):
    q1 = Factor(A)
    hospital_types_kor = ['종합병원', '병원', '요양병원', '한방병원', '정신병원']  # 한글 명칭
    hospital_types_eng = ['General hospital', 'Hospital', 'Convalescent hospita', 'Oriental medicine hospital', 'Psychiatric hospital']  # 영어 명칭
    colors = ['cyan', 'red', 'yellow', 'gray', 'blue', 'purple']
    alphabet_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 알파벳 라벨

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))  # 2행 3열의 서브플롯 생성
    axes = axes.flatten()  # 2D 배열을 1D로 변환하여 쉽게 반복문으로 접근 가능

    # 전체 병원의 데이터를 좌측 상단에 출력
    total_data = q1['USE_QTY_kWh']
    total_count = len(total_data)  # 전체 데이터 개수 계산
    sns.histplot(total_data, kde=True, color='green', alpha=0.8, bins=50, label='Total', ax=axes[0])

    # KDE 라인의 색상을 검정으로 변경
    for line in axes[0].lines:
        line.set_color('black')

    # 제목에 영어 명칭과 데이터 개수를 포함
    axes[0].set_title(f"(a) Total hospital (n={total_count})", fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Total annual energy (kWh/yr)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].tick_params(axis='x', labelsize=12)  # xticks 글자 크기 조정
    axes[0].tick_params(axis='y', labelsize=12)  # yticks 글자 크기 조정

    # 나머지 병원 유형별 데이터를 각 서브플롯에 출력
    for i, (hospital_kor, hospital_eng, color) in enumerate(zip(hospital_types_kor, hospital_types_eng, colors), start=1):
        Group_1 = q1[q1['종별코드명'].isin([hospital_kor])]
        count = len(Group_1)  # 데이터 개수 계산
        sns.histplot(Group_1['USE_QTY_kWh'], kde=True, color=color, alpha=0.8, bins=50, label=hospital_eng, ax=axes[i])

        # KDE 라인의 색상을 검정으로 변경
        for line in axes[i].lines:
            line.set_color('black')

        # 제목에 영어 명칭과 데이터 개수를 포함
        axes[i].set_title(f"{alphabet_labels[i]} {hospital_eng} (n={count})", fontsize=16, fontweight='bold')
        axes[i].set_xlabel('Total annual energy (kWh/yr)', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].tick_params(axis='x', labelsize=12)  # xticks 글자 크기 조정
        axes[i].tick_params(axis='y', labelsize=12)  # yticks 글자 크기 조정

    plt.tight_layout()
    plt.show()

Factor2_separate(q1)



def Factor2_grouped(A):
    q1 = Factor(A)

    # 병원 그룹 정의
    hospital_groups = {
        'Group 1: 종합병원': ['종합병원'],
        'Group 2: 병원, 요양병원': ['병원', '요양병원'],
        'Group 3: 한방병원, 정신병원, 치과병원': ['한방병원', '정신병원', '치과병원']
    }

    colors = ['cyan', 'red', 'blue']  # 각 그룹에 대한 색상 지정

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # 1행 3열의 서브플롯 생성
    axes = axes.flatten()  # 2D 배열을 1D로 변환하여 쉽게 반복문으로 접근 가능

    for i, (group_name, hospitals) in enumerate(hospital_groups.items()):
        Group_1 = q1[q1['종별코드명'].isin(hospitals)]
        sns.histplot(Group_1['USE_QTY_kWh'], kde=True, color=colors[i], alpha=0.8, bins=50, label=group_name,
                     ax=axes[i])
        axes[i].set_title(group_name)
        axes[i].set_xlabel('USE_QTY_kWh')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


Factor2_grouped(q1)


def Factor2_grouped_with_individuals(A):
    q1 = Factor(A)

    # 병원 그룹 정의
    hospital_groups = {
        'Group 1: 종합병원': ['종합병원'],
        'Group 2: 병원, 요양병원': ['병원', '요양병원'],
        'Group 3: 한방병원, 정신병원, 치과병원': ['한방병원', '정신병원', '치과병원']
    }

    group_colors = ['cyan', 'red', 'blue']  # 각 그룹에 대한 색상 지정
    individual_colors = ['cyan', 'red', 'yellow', 'gray', 'purple', 'green']  # 개별 병원 색상

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # 1행 3열의 서브플롯 생성
    axes = axes.flatten()  # 2D 배열을 1D로 변환하여 쉽게 반복문으로 접근 가능

    for i, (group_name, hospitals) in enumerate(hospital_groups.items()):
        # 그룹 전체 데이터를 합친 히스토그램
        Group_1 = q1[q1['종별코드명'].isin(hospitals)]
        sns.histplot(Group_1['USE_QTY_kWh'], kde=True, color=group_colors[i], alpha=0.5, bins=50,
                     label="Total", ax=axes[i])

        # 그룹 내 개별 병원들의 히스토그램을 같은 플롯에 추가
        for j, hospital in enumerate(hospitals):
            individual_data = q1[q1['종별코드명'] == hospital]
            sns.histplot(individual_data['USE_QTY_kWh'], color=individual_colors[j], alpha=0.8, bins=50,
                         label=hospital, ax=axes[i])

        axes[i].set_title(group_name)
        axes[i].set_xlabel('USE_QTY_kWh')
        axes[i].set_ylabel('Frequency')
        axes[i].legend(title='병원 종류')

    plt.tight_layout()
    plt.show()


Factor2_grouped_with_individuals(q1)

# q1=Factor(q1)
# # q1 = q1.sample(frac=1).reset_index(drop=True)
# # Case A.
# Group_1 = q1[q1['종별코드명'].isin(['종합병원'])]
# Group_1=Group_1[['연면적','용적률산정연면적', '대지면적', '층수','지하층수', '지상층수','승강기수', '의사수', '병상수','USE_QTY_kWh']]
#
# Group_1 = IQR_fitter(Group_1)
#
# # Group_1 = transform_hos(Group_1)
# print('# of data1 : ',len(Group_1))
# print(Group_1)
#
#
#
# stats_summary = pd.DataFrame()
#
# stats_summary['N'] = Group_1.count()
# stats_summary['Min'] = Group_1.min()
# stats_summary['Max'] = Group_1.max()
# stats_summary['Mean'] = Group_1.mean()
# stats_summary['Std. Deviation'] = Group_1.std()
# # stats_summary['Skewness'] = Group_1.apply(skew)
# # stats_summary['Skewness Std. Error'] = np.sqrt(6/Group_1.count())
# # stats_summary['Kurtosis'] = Group_1.apply(kurtosis)
# # stats_summary['Kurtosis Std. Error'] = np.sqrt(24/Group_1.count())
# print(stats_summary)
# filename1 = "Group1.xlsx"
# file_path1 = os.path.join(dir_path, filename1)
# stats_summary.to_excel(file_path1)
#
# #
# # ax = sns.histplot(Group_1['USE_QTY_kWh'], kde= True, color = 'Cyan', alpha = 0.8, bins=50, label='종합병원')
# # ax.lines[0].set_color('Black')
# # ax.lines[0].set_linewidth(3)       # Make the KDE line thicker
#
#
# Group_2 = q1[q1['종별코드명'].isin(['병원', '요양병원'])]
# Group_2 = IQR_fitter(Group_2)
#
# # Group_2 = transform_hos(Group_2)
# print('# of data2 : ',len(Group_2['종별코드명']))
# Group_2_병원 = Group_2[Group_2['종별코드명'].isin(['병원'])]
# Group_2_요양병원 = Group_2[Group_2['종별코드명'].isin(['요양병원'])]
# # #
# # ax = sns.histplot(Group_2['USE_QTY_kWh'], kde= True, color = 'cyan', alpha = 0.8, bins=50, label='Total')
# # ax.lines[0].set_color('black')
# # ax.lines[0].set_linewidth(3)       # Make the KDE line thicker
# # plt.hist(Group_2_요양병원['USE_QTY_kWh'], color = 'red', alpha = 0.8, bins=50, label='요양병원')
# # plt.hist(Group_2_병원['USE_QTY_kWh'], color = 'green', alpha = 0.8, bins=50, label='병원')#, hatch='-'
#
#
#
# Group_2=Group_2[['연면적','용적률산정연면적', '대지면적', '층수','지하층수', '지상층수','승강기수', '의사수', '병상수','USE_QTY_kWh']]
#
#
# stats_summary = pd.DataFrame()
#
# stats_summary['N'] = Group_2.count()
# stats_summary['Min'] = Group_2.min()
# stats_summary['Max'] = Group_2.max()
# stats_summary['Mean'] = Group_2.mean()
# stats_summary['Std. Deviation'] = Group_2.std()
# # stats_summary['Skewness'] = Group_2.apply(skew)
# # stats_summary['Skewness Std. Error'] = np.sqrt(6/Group_2.count())
# # stats_summary['Kurtosis'] = Group_2.apply(kurtosis)
# # stats_summary['Kurtosis Std. Error'] = np.sqrt(24/Group_2.count())
# print(stats_summary)
# filename2 = "Group2.xlsx"
# file_path2 = os.path.join(dir_path, filename2)
# stats_summary.to_excel(file_path2)
#
#
#
#
#
#
# # Group_3 = q1[q1['종별코드명'].isin(['요양병원'])]
# # # Group_3 = IQR_fitter(Group_3)
# # # Group_3 = transform_hos(Group_3)
# # print("# of PK3 : ", Group_3['매칭표제부PK'].nunique())
# # print('# of data3 : ',len(Group_3))
#
#
# Group_4 = q1[q1['종별코드명'].isin(['한방병원','정신병원','치과병원'])]
# Group_4 = IQR_fitter(Group_4)
#
#
# # Group_4 = transform_hos(Group_4)
# print('# of data4 : ',len(Group_4))
# Group_4_한방병원 = Group_4[Group_4['종별코드명'].isin(['한방병원'])]
# Group_4_정신병원= Group_4[Group_4['종별코드명'].isin(['정신병원'])]
# Group_4_치과병원= Group_4[Group_4['종별코드명'].isin(['치과병원'])]
# #
# #
# #
# #
# ax = sns.histplot(Group_4['USE_QTY_kWh'], kde= True, color = 'Cyan', alpha = 0.8, bins=50, label='Total')
# ax.lines[0].set_color('Black')
# ax.lines[0].set_linewidth(3)       # Make the KDE line thicker
# plt.hist(Group_4_한방병원['USE_QTY_kWh'], color = 'red', alpha = 0.8, bins=50, label='한방병원')
# plt.hist(Group_4_정신병원['USE_QTY_kWh'], color = 'green', alpha = 0.8, bins=50, label='정신병원')
# plt.hist(Group_4_치과병원['USE_QTY_kWh'], color = 'blue', alpha = 0.8, bins=50, label='치과병원')
#
#
#
# Group_4=Group_4[['연면적','용적률산정연면적', '대지면적', '층수','지하층수', '지상층수','승강기수', '의사수', '병상수','USE_QTY_kWh']]
#
# stats_summary = pd.DataFrame()
#
# stats_summary['N'] = Group_4.count()
# stats_summary['Min'] = Group_4.min()
# stats_summary['Max'] = Group_4.max()
# stats_summary['Mean'] = Group_4.mean()
# stats_summary['Std. Deviation'] = Group_4.std()
# # stats_summary['Skewness'] = Group_4.apply(skew)
# # stats_summary['Skewness Std. Error'] = np.sqrt(6/Group_4.count())
# # stats_summary['Kurtosis'] = Group_4.apply(kurtosis)
# # stats_summary['Kurtosis Std. Error'] = np.sqrt(24/Group_4.count())
# print(stats_summary)
# filename3 = "Group3.xlsx"
# file_path3 = os.path.join(dir_path, filename3)
# stats_summary.to_excel(file_path3)
#
#
# # plt.hist(hos_7['USE_QTY_kWh'], color = 'black', alpha = 0.4, bins=50, label='종합병원 제외')
# plt.legend()
# plt.xlabel('USE_QTY_kWh')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.show()
