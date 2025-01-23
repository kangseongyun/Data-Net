# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')


def data_preprocessing(load_f):
    df=pd.read_excel(load_f)
    df['EUI(연면적)']=df['USE_QTY_kWh']/df['연면적(㎡)']

    df['EUI(용적률산정연면적)']=df['USE_QTY_kWh']/df['용적률산정연면적(㎡)']

    df['USE_QTY_kWh(비율)']=df['USE_QTY_kWh']*df['주용도(의료시설) 비율(%)']

    df['EUI(연면적)(비율)']=df['EUI(연면적)']*df['주용도(의료시설) 비율(%)']

    df['EUI(용적률산정연면적)(비율)']=df['EUI(용적률산정연면적)']*df['주용도(의료시설) 비율(%)']



    A=['USE_QTY_kWh','EUI(연면적)','EUI(용적률산정연면적)','USE_QTY_kWh(비율)','EUI(연면적)(비율)','EUI(용적률산정연면적)(비율)','주용도(의료시설) 비율(%)','electricity_kWh','gas_kWh','districtheat_kWh','대지면적(㎡)','건축면적(㎡)','건폐율(%)','연면적(㎡)','용적률산정연면적(㎡)','용적률(%)','높이(m)','지상층수','지하층수','승용승강기수','비상용승강기수','부속건축물수','부속건축물면적(㎡)','총동연면적(㎡)']
    B=['총의사수','의과일반의 인원수','의과인턴 인원수','의과레지던트 인원수','의과전문의 인원수','치과일반의 인원수','치과인턴 인원수','치과레지던트 인원수','치과전문의 인원수','한방일반의 인원수','한방인턴 인원수','한방레지던트 인원수','한방전문의 인원수']
    # df.dropna(subset=df.columns[130:143], how='all', inplace=True)
    print('세부정보0')
    print("# of PK : ", df['매칭표제부PK'].nunique())
    print('# of data : ', len(df))
    print(' ')

    if "시설정보" in load_f:
        C=['일반입원실상급병상수','일반입원실일반병상수','성인중환자병상수','소아중환자병상수','신생아중환자병상수','정신과폐쇄상급병상수','정신과폐쇄일반병상수','격리병실병상수','무균치료실병상수','분만실병상수','수술실병상수','응급실병상수','물리치료실병상수']

        # df=df[df['총의사수'] > 0]
        # df=df[df['의과전문의 인원수'] > 0]

        columns_to_use = A + B + C

    if "세부정보" in load_f:
        C=['진료시간']
        columns_to_use = A + B + C

    if "진료과목정보" in load_f:

        C=df.columns[120:].tolist()
        print(C)
        columns_to_use = A + B + C

    if "의료장비정보" in load_f:
        C = ['CT', 'MRI', '골밀도검사기', '양전자단층촬영기 (PET)', '유방촬영장치',
                          '종양치료기 (Cyber Knife)', '종양치료기 (Gamma Knife)', '종양치료기 (양성자치료기)',
                          '체외충격파쇄석기', '초음파영상진단기', '콘빔CT', '혈액투석을위한인공신장기']

        columns_to_use = A + B + C


    df.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\test..xlsx")
    # df = df[~df['CT'].isna()]  ## '단위코드'열에 모두 value가 존재하므로 의미가 없음.

    # corr = df[columns_to_use].corr(method='pearson').sort_values(by='USE_QTY_kWh', ascending=True)
    corr = df[columns_to_use].corr(method='pearson')

    plt.figure(figsize=(12, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"shrink": .8}, linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 7})
    # sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={"shrink": .8}, linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 7})

    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    return corr



# w1=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_2_2018_01_시설정보.xlsx")
# w2=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_2_2019_01_시설정보.xlsx")
# w3=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_2_2020_01_시설정보.xlsx")
# w4=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_2_2021_01_시설정보.xlsx")
# w5=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_2_2022_01_시설정보.xlsx")


w1=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_2_2018_02_세부정보.xlsx")
w2=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_2_2019_02_세부정보.xlsx")
w3=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_2_2020_02_세부정보.xlsx")
w4=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_2_2021_02_세부정보.xlsx")
w5=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_2_2022_02_세부정보.xlsx")

# w1=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_2_2018_03_진료과목정보.xlsx")
# w2=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_2_2019_03_진료과목정보.xlsx")
# w3=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_2_2020_03_진료과목정보.xlsx")
# w4=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_2_2021_03_진료과목정보.xlsx")
# w5=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_2_2022_03_진료과목정보.xlsx")

# w1=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_2_2018_03_진료과목정보.xlsx")
# w2=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_2_2019_03_진료과목정보.xlsx")
# w3=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_2_2020_03_진료과목정보.xlsx")
# w4=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_2_2021_03_진료과목정보.xlsx")
# w5=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_2_2022_03_진료과목정보.xlsx")

# w1=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_2_2018_05_의료장비정보.xlsx")
# w2=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_2_2019_05_의료장비정보.xlsx")
# w3=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_2_2020_05_의료장비정보.xlsx")
# w4=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_2_2021_05_의료장비정보.xlsx")
# w5=data_preprocessing(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_2_2022_05_의료장비정보.xlsx")

w1.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\상관계수_2018.xlsx")
w2.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\상관계수_2019.xlsx")
w3.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\상관계수_2020.xlsx")
w4.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\상관계수_2021.xlsx")
w5.to_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\상관계수_2022.xlsx")

# 세부정보_2018 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_02_세부정보.xlsx")
# 세부정보_2019 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_02_세부정보.xlsx")
# 세부정보_2020 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_02_세부정보.xlsx")
# 세부정보_2021 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_02_세부정보.xlsx")
# 세부정보_2022 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_02_세부정보.xlsx")

# 진료과목정보_2018 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_03_진료과목정보.xlsx")
# 진료과목정보_2019 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_03_진료과목정보.xlsx")
# 진료과목정보_2020 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_03_진료과목정보.xlsx")
# 진료과목정보_2021 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_03_진료과목정보.xlsx")
# 진료과목정보_2022 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_03_진료과목정보.xlsx")

# 의료장비정보_2018 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_05_의료장비정보.xlsx")
# 의료장비정보_2019 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_05_의료장비정보.xlsx")
# 의료장비정보_2020 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_05_의료장비정보.xlsx")
# 의료장비정보_2021 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_05_의료장비정보.xlsx")
# 의료장비정보_2022 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_05_의료장비정보.xlsx")

# 식대가산정보_2018 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_06_식대가산정보.xlsx")
# 식대가산정보_2019 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_06_식대가산정보.xlsx")
# 식대가산정보_2020 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_06_식대가산정보.xlsx")
# 식대가산정보_2021 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_06_식대가산정보.xlsx")
# 식대가산정보_2022 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_06_식대가산정보.xlsx")

# 간호등급정보_2018 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_07_간호등급정보.xlsx")
# 간호등급정보_2019 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_07_간호등급정보.xlsx")
# 간호등급정보_2020 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_07_간호등급정보.xlsx")
# 간호등급정보_2021 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_07_간호등급정보.xlsx")
# 간호등급정보_2022 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_07_간호등급정보.xlsx")

# 특수진료정보_2018 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018\데이터넷_1_2018_08_특수진료정보.xlsx")
# 특수진료정보_2019 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019\데이터넷_1_2019_08_특수진료정보.xlsx")
# 특수진료정보_2020 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020\데이터넷_1_2020_08_특수진료정보.xlsx")
# 특수진료정보_2021 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021\데이터넷_1_2021_08_특수진료정보.xlsx")
# 특수진료정보_2022 = pd.read_excel(r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022\데이터넷_1_2022_08_특수진료정보.xlsx")
