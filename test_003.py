import dataset.marcap_utils as mu
import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6    # matlab 차트의 기본 크기를 15,6으로 지정해 줍니다.


df_k200 = mu.marcap_date_range('2017-01-01', '2018-12-31', mu.krx200_code_list)
# print(df_k200.head(400))
print(df_k200.shape)
# df_pv = pd.pivot_table(df_k200, index='Date', columns='Code', values='Marcap')
df_pv = pd.pivot_table(df_k200, index='Code', columns='Date', values='Marcap')
print(df_pv.shape)
print(df_pv.head(400))

# # dfValues = df_pv['2017-01-02'].values
# dfValues = df_pv['000020'].values
# print(dfValues)
# print(type(dfValues))

dfValues = df_pv['2017-01-02'].values
print(dfValues)
print(type(dfValues))


# df_a = mu.marcap_date_range_dateindexed('1995', '2018', '009150')
# plt.plot(df_a['Close']*df_a['Stocks'])
# plt.show(block=True)
# df_a = mu.marcap_date_range_dateindexed('2017', '2018', ['009150', '005930'])
# print(df_a.head(10))
#
# df_a = mu.marcap_date_range_dateindexed('2017', '2018', '009150')
# print(df_a.head(10))
# plt.plot(df_a['Close']*df_a['Stocks'])
# plt.show()

# code = '009150'
# print(type(code))
#
# code = ['009150', '005930']
# print(type(code))
#
# if isinstance(code, list):
#     print('list')
# else:
#     print('??')

