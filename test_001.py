import dataset.marcap_utils as mu
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6    # matlab 차트의 기본 크기를 15,6으로 지정해 줍니다.
import pandas as pd
# df = mu.marcap_date_range('2017-12-15', '2018-01-15', code='009150')
# print(df.columns)

# df = mu.marcap_date_range('1995-01-01', '2018-12-31')
# print(df.describe())


#
# code = '005930'  # 삼성전자
# # code = '009150'  # 삼성전기
# df_stock = mu.marcap_date_range('2017-01-01', '2018-12-31', code)
# print(df_stock.head())
# # df_stock.reindex('Date')
#
# print(df_stock.index)
# plt.plot(df_stock['Close']*df_stock['Stocks'])
# plt.show()
# # df_stock['close'].plot(figsize=(16, 6))
# #
#


# df = mu.marcap_date_range('2014-01-01', '2018-12-31', code='009150')
# ts = df[['Close', 'Date']]
# print(ts[ts['Date'] >= '20180101'].index)
# print(ts[982:])
# print(ts['Date'] >= '20180101')
# print(ts.head())


train_start_date = '2017-01-01'
# train_end_date = '2017-01-31'
train_end_date = '2018-06-30'
test_start_date = '2018-06-18'
test_end_date = '2018-12-31'
x_window_length = 10
# dt_train_end_date = pd.to_datetime(train_end_date)
df = mu.marcap_date_range(train_start_date, train_end_date, code='009150')
date_list = df['Date'].unique()
print(date_list)
total_x = len(date_list) - x_window_length
print(total_x)

# train_data = [([ for j in range(x_window_length)], df.iat[i+5,1]) for i in range(115)]
# train_date_list = [ date for date in date_list if date <= dt_train_end_date]
# print(train_date_list)

print(df[(df['Date'] == date_list[0]) & (df['Code'] == '005930')])


train_data = [([df.at[i+j,'Close'] for j in range(x_window_length)], df.at[i+x_window_length,'Close']) for i in range(total_x)]
print(train_data)
for i in range(10):
    print(df.at[i,'Close'])

print(df.head(10))

'''print(df['Date'].unique())
# print(type(date_list))
print(df['Code'].unique())

# dt_train_end_date = pd.to_datetime(train_end_date)
# print(date_list[date_list<=dt_train_end_date])

print(df[(df['Date'] == date_list[0]) & (df['Code'] == '005930')])
print(df[(df['Date'] == date_list[0]) & (df['Code'] == '005930')]['Close'])
# train_start_date

# df.at[20, 'Month']

train_data = [([df.iat[i+j,1] for j in range(5)], df.iat[i+5,1]) for idx, date in enumerate(date_list) where date ]'''