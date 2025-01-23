import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')


dir_path = r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\병합본"
filename = "데이터넷_1_01_시설정보.xlsx"
file_path = os.path.join(dir_path, filename)
q1 = pd.read_excel(file_path)
q1 = q1[q1['year_use'].isin([2022])]


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



def Factor2_separate(A):
    q1 = Factor(A)
    hospital_types_kor = ['종합병원', '병원', '요양병원', '한방병원', '정신병원', '치과병원']  # 한글 명칭
    hospital_types_eng = ['General Hospital', 'Hospital', 'Convalescent hospita', 'Oriental Medicine Hospital', 'Psychiatric Hospital', 'Dental Hospital']  # 영어 명칭
    colors = ['cyan', 'red', 'yellow', 'gray', 'blue', 'purple']
    alphabet_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 알파벳 라벨

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 8))  # 2행 3열의 서브플롯 생성
    axes = axes.flatten()  # 2D 배열을 1D로 변환하여 쉽게 반복문으로 접근 가능

    for i, (hospital_kor, hospital_eng, color) in enumerate(zip(hospital_types_kor, hospital_types_kor, colors)):
        Group_1 = q1[q1['종별코드명'].isin([hospital_kor])]
        count = len(Group_1)  # 데이터 개수 계산
        sns.histplot(Group_1['USE_QTY_kWh'], kde=True, color=color, alpha=0.8, bins=50, label=hospital_eng, ax=axes[i])

        # KDE 라인의 색상을 검정으로 변경
        for line in axes[i].lines:
            line.set_color('black')

        # 제목에 영어 명칭과 데이터 개수를 포함
        axes[i].set_title(f"{alphabet_labels[i]} {hospital_eng} (n={count})", fontsize=25,fontweight='bold')
        axes[i].set_xlabel('Total annual energy (kWh/yr)', fontsize=20)
        axes[i].set_ylabel('Frequency', fontsize=20)
        axes[i].tick_params(axis='x', labelsize=20)  # xticks 글자 크기 조정
        axes[i].tick_params(axis='y', labelsize=20)  # yticks 글자 크기 조정
    plt.tight_layout()
    plt.savefig("병원종별 연간 에너지사용량 histogram" + '.tiff', format='tiff', dpi=300)

    plt.show()

Factor2_separate(q1)




