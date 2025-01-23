import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')

# categories = ['Total hospital', 'General hospital', 'Hospital', 'Convalescent hospital', 'Oriental Medicine hospital', 'Psychiatric hospital']
# variables = ['Gross floor area', 'Number of doctors', 'Number of elevators', 'Number of beds', 'Total number of floors']


categories = ['전체 병원', '종합병원', '병원+치과병원', '요양병원', '한방병원', '정신병원']
variables = ['연면적', '의사수', '승강기수', '병상수', '전체층수']


beta_values = {
    '연면적': [0.79, 0.95, 0.42, 0.50, 0.54, 0],
    '의사수': [0.40, 0.39, 0.34, 0.46, 0.44, 0.64],
    '승강기수': [-0.17, -0.35, 0.16, 0, 0, 0],
    '병상수': [-0.04, 0, 0.11, -0.15, 0, 0],
    '전체층수': [0, 0, 0, 0.10, 0, 0]
}


x = np.arange(len(categories))
width = 0.15
r_squared_values = [0.91, 0.92, 0.66, 0.60, 0.89, 0.41]

fig, ax1 = plt.subplots(figsize=(20, 8))

bars = []
for i, variable in enumerate(variables):
    bars.append(ax1.bar(x + i*width, beta_values[variable], width, label=variable))

# ax1.set_xlabel('Hospital Types', fontsize=18)
ax1.set_ylabel('Standardized Regression Coefficient (Beta)', fontsize=25)
# ax1.set_title('Standardized Regression Coefficients by Hospital Type with R-squared', fontsize=20)
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels(categories, rotation=0, fontsize=25)

ax1.tick_params(axis='x', labelsize=25)
ax1.tick_params(axis='y', labelsize=25)

ax1.axhline(0, color='black', linewidth=1)

ax1.set_ylim(-0.4, 1.0)

ax2 = ax1.twinx()
r_squared_line, = ax2.plot(x + width * 2, r_squared_values, color='red', marker='o', markersize=15, linestyle='-', linewidth=4, label='R-squared')
ax2.set_ylabel('R-squared', fontsize=25)
ax2.tick_params(axis='y', labelsize=25)
ax2.set_ylim(0, 1.0)
ax2.axhline(0.6, color='#FF9999', linestyle='--', linewidth=4, label='y=0.6')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = [r_squared_line], ['R-squared']
all_handles = handles1 + handles2
all_labels = labels1 + labels2

# ax1.legend(all_handles, all_labels, fontsize=25, loc='upper right')
# ax1.legend(all_handles, all_labels, fontsize=22, loc='lower right', ncol=len(all_labels))
# ax1.legend(all_handles, all_labels, fontsize=25, loc='lower right', ncol=3)
ax1.legend(all_handles, all_labels, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1)

plt.tight_layout()
plt.savefig("단계적 회귀 결과 요약" + '.tiff', format='tiff', dpi=300)

plt.show()
