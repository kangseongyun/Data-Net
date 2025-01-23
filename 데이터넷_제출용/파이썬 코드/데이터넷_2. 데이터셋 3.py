import os
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family= 'Malgun Gothic')


##################################### 데이터셋3. 건축물대장_층별개요  ##########################################################################################################

### step 1. 층별개요 데이터 input ########################################################################################

## 데이터 경로 입력
dir_path =r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)"


## 층별개요 반영
filenames_bldg="데이터넷_의료시설(건축물대장).xlsx"
file_path = os.path.join(dir_path, filenames_bldg)
df_bldg2 = pd.read_excel(file_path, sheet_name='층별개요')

print('BEFORE : 층별개요_건축물대장')
print("# of PK1 : ", df_bldg2['매칭표제부PK'].nunique())  # of PK1 :  7581
print('# of data1 : ', df_bldg2.shape[0])               # of data1 :  57007
print(' ')



### step 2. 데이터셋1+2 결합본에 있는 매칭표제부PK만 사용 ####################################################################

# filename = "데이터넷_데이터셋(1+2) 결합본.xlsx"
# file_path = os.path.join(dir_path,filename)
# hos_00 = pd.read_excel(file_path)


def caculate_rate(df_bldg2,input_path,result_path):
    hos_00 = pd.read_excel(input_path)
    PKlist=hos_00['매칭표제부PK'].unique()
    df_bldg2=df_bldg2[df_bldg2['매칭표제부PK'].isin(PKlist)]

    print('Afer : 층별개요_건축물대장')
    print("# of PK2 : ", df_bldg2['매칭표제부PK'].nunique())   # of PK2 :  1429
    print('# of data2 : ', df_bldg2.shape[0])                # of data2 :  15854
    print(' ')



    ### 참고용. 주용도코드명 체크  ############################################################################################

    print('참고용: 주용도코드명 체크')
    print("# of 주용도코드명 : ", df_bldg2['주용도코드명'].nunique())   # of 주용도코드명 :  105 (주용도코드명=nan/면적(㎡)=0인 것 포함: 추후 영향을 주기 때문)
    print('# of data : ', df_bldg2.shape[0])                         # of data :  15854 (주용도코드명=nan: 9, 면적(㎡)=0: 35)
    print(' ')
    print(df_bldg2['주용도코드명'].unique().tolist())
    print(' ')
    print(' ')

    # # ['병원', '기타제1종근린생활시설', '기타제2종근린생활시설', '요양병원', '기숙사', '연립주택', '노인복지시설', '단독주택', '치과병원', '의원', '소매점', '휴게음식점', '주차장', '장례식장',
    # # '종합병원', '일반음식점', '요양소', '사무소', '의료기기판매소', '기타병원', '의료시설', '기타의료시설', '통신용시설', '기타공연장', '일반공장', '골프연습장', '학원', '한방병원', '미용원',
    # # '창고', '조산원', '산부인과병원', '조산소', '체력단련장', '부동산중개사무소', '정신병원', '제과점', '한의원', '기타사무소', '기타일반업무시설', '목욕장', '기타교육연구시설', '볼링장',
    # # '직업훈련소', '금융업소', '부대시설', '제조업소', '기타공공시설', '세탁소', '이용원', '노래연습장', '기타근린생활시설', '기원', '멀티미디어문화컨텐츠설비제공업소', '유흥주점', '사회복지시설',
    # # '치과의원', '회의장', '수녀원', '단란주점', '기타창고시설', '기타노유자시설', '관광호텔', '정수장', '산후조리원', '이(미)용원', '교회', '안마시술소', '교습소', '일반목욕장', '에어로빅장',
    # # '안마원', '수리점', '당구장', '기타 운동시설', '연구소', '게임제공업소', '인터넷컴퓨터게임시설제공업소', '독서실', '기타발전시설', '동물병원', '기타판매시설', '자동차영업소', '오피스텔',
    # # '상점', '다가구주택', '의약품판매소', '서점(1종근.생미해당)', '일반창고', '비디오물감상실', '체육도장', '방송국', '기타전시장', '여관', '교육(연수)원', '체육장', '공공시설', '일반업무시설',
    # # '기타공장', '기타묘지관련시설', '어린이집', '탁구장', '음악당', '제1종근린생활시설','NaN']



    ##################################### 주용도코드명별 갯수 graph ##########################################################################################################

    df2 = df_bldg2['주용도코드명'].value_counts()

    # 전체 주용도코드명별 갯수 산정
    filenames_bldg="데이터넷_주용도코드명별 갯수.xlsx"
    file_path = os.path.join(dir_path, filenames_bldg)
    df2.to_excel(file_path)

    bottom_categories = df2[-85:]  # 주용도코드명별 하위 85개
    top_categories = df2[:-85]  # 주용도코드명별 상위 19개

    others_count = bottom_categories.sum() # 주용도코드명별 하위 85개에 대한 갯수 합산

    new_value_counts = pd.concat([top_categories, pd.Series({'Others': others_count})]) # 데이터 병합

    plt.figure(figsize=(10, 6))
    new_value_counts.plot(kind='bar', color='skyblue')  # You can choose any color
    plt.title('Count of 주용도코드명')
    plt.xlabel('주용도코드명')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



    ##################################### 주용도 비율 산정(고려대 주용도 혼합도와 다름) ##########################################################################################

    df_bldg2_group=df_bldg2.pivot_table(index='매칭표제부PK', columns='주용도코드명',values='면적(㎡)',aggfunc='sum')

    hos_list=['병원','요양병원','종합병원','기타병원','의료시설','치과병원','한방병원','정신병원','산부인과병원', '기타의료시설', '의원', '조산원', '한의원', '치과의원','산후조리원','조산소']
    # ※참고 어디까지를 의료시설로 봐야 할지 고민 ex)요양병원의 노인복지시설, 종합병원의 장례식장
    # 앞선 데이터셋2의 종별코드명에서 6개의 병원type에 대해 암호화요양코드 1개당 매칭표제부 1개를 매칭했기 때문에 병원에 해당되는 용도인 의원까지 지정함.
    existing_hos_list = [col for col in hos_list if col in df_bldg2_group.columns]

    hospital_related = df_bldg2_group[existing_hos_list].sum(axis=1)

    # 병원 관련 열들을 제외한 나머지 열을 합침
    other = df_bldg2_group.drop(columns=existing_hos_list).sum(axis=1)

    # 결과를 새로운 데이터프레임으로 만듦
    result = pd.DataFrame({
        '병원 면적': hospital_related,
        '기타 면적': other
    }, index=df_bldg2_group.index)

    result['면적 합계'] = result.sum(axis=1)

    result_ratio = pd.DataFrame()
    result_ratio['주용도(의료시설) 비율(%)'] = result['병원 면적']*100/result['면적 합계']
    result_ratio['기타 면적 비율(%)'] = result['기타 면적']*100/result['면적 합계']



    ### 주용도비율 CDF(Cumulative Distribution Function) graph 출력 ##########################################################

    # 데이터를 정렬하고, 누적 합을 계산
    df_sorted = result_ratio.sort_values('주용도(의료시설) 비율(%)')
    df_sorted['cumulative'] = df_sorted['주용도(의료시설) 비율(%)'].cumsum() / df_sorted['주용도(의료시설) 비율(%)'].sum()
    df_sorted.reset_index(inplace=True)

    # print(df_sorted.head())
    print(' ')
    print("# of PK : ", df_sorted['매칭표제부PK'].nunique()) # of PK :  1429
    print('# of data : ', df_sorted.shape[0])              # of data :  1429
    print(' ')

    # 누적 분포 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['주용도(의료시설) 비율(%)'], df_sorted['cumulative'], marker='o', linestyle='-')
    plt.title('Cumulative Distribution Graph')
    plt.xlabel('주용도(의료시설) 비율(%)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    ##################################### 데이터셋(1+2) + 데이터셋3 병합 ##########################################################################################################

    df_sorted_merge1 = pd.merge(left=df_sorted, right=hos_00, how='inner', on=['매칭표제부PK'])
    print(df_sorted_merge1.head())
    print(' ')

    print('데이터셋(1+2) + 데이터셋3 병합')
    print("# of PK : ", df_sorted_merge1['매칭표제부PK'].nunique()) # of PK :  1429
    print('# of data : ', df_sorted_merge1.shape[0])              # of data :  11418
    print(' ')

    filenames_bldg=result_path
    file_path = os.path.join(filenames_bldg)
    df_sorted_merge1.to_excel(file_path)

dir_path_2018=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2018"
dir_path_2019=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2019"
dir_path_2020=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2020"
dir_path_2021=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2021"
dir_path_2022=r"C:\Users\user\Desktop\데이터넷_의료시설(건축물대장,에너지사용량)\2022"

filenames_2018="데이터넷_1_2018_02_세부정보.xlsx"
filenames_2019="데이터넷_1_2019_02_세부정보.xlsx"
filenames_2020="데이터넷_1_2020_02_세부정보.xlsx"
filenames_2021="데이터넷_1_2021_02_세부정보.xlsx"
filenames_2022="데이터넷_1_2022_02_세부정보.xlsx"

# filenames_2018="데이터넷_1_2018_03_진료과목정보.xlsx"
# filenames_2019="데이터넷_1_2019_03_진료과목정보.xlsx"
# filenames_2020="데이터넷_1_2020_03_진료과목정보.xlsx"
# filenames_2021="데이터넷_1_2021_03_진료과목정보.xlsx"
# filenames_2022="데이터넷_1_2022_03_진료과목정보.xlsx"

# filenames_2018="데이터넷_1_2018_05_의료장비정보.xlsx"
# filenames_2019="데이터넷_1_2019_05_의료장비정보.xlsx"
# filenames_2020="데이터넷_1_2020_05_의료장비정보.xlsx"
# filenames_2021="데이터넷_1_2021_05_의료장비정보.xlsx"
# filenames_2022="데이터넷_1_2022_05_의료장비정보.xlsx"

file_path_2018 = os.path.join(dir_path_2018, filenames_2018)
file_path_2019 = os.path.join(dir_path_2019, filenames_2019)
file_path_2020 = os.path.join(dir_path_2020, filenames_2020)
file_path_2021 = os.path.join(dir_path_2021, filenames_2021)
file_path_2022 = os.path.join(dir_path_2022, filenames_2022)

filenames_2018="데이터넷_2_2018_02_세부정보.xlsx"
filenames_2019="데이터넷_2_2019_02_세부정보.xlsx"
filenames_2020="데이터넷_2_2020_02_세부정보.xlsx"
filenames_2021="데이터넷_2_2021_02_세부정보.xlsx"
filenames_2022="데이터넷_2_2022_02_세부정보.xlsx"

file_result_2018 = os.path.join(dir_path_2018, filenames_2018)
file_result_2019 = os.path.join(dir_path_2019, filenames_2019)
file_result_2020 = os.path.join(dir_path_2020, filenames_2020)
file_result_2021 = os.path.join(dir_path_2021, filenames_2021)
file_result_2022 = os.path.join(dir_path_2022, filenames_2022)


caculate_rate(df_bldg2,file_path_2018,file_result_2018)
caculate_rate(df_bldg2,file_path_2019,file_result_2019)
caculate_rate(df_bldg2,file_path_2020,file_result_2020)
caculate_rate(df_bldg2,file_path_2021,file_result_2021)
caculate_rate(df_bldg2,file_path_2022,file_result_2022)