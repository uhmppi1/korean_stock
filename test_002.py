import dataset.marcap_utils as mu

# df = mu.marcap_date_range('2017-01-01', '2018-12-31', code='009150')
train_start_date = '2017-01-01'
train_end_date = '2018-06-30'
test_start_date = '2018-07-01'
test_end_date = '2018-12-31'
x_window_length = 10
code = '009150'
df = mu.marcap_date_range(train_start_date, test_end_date, code)
date_list = df['Date'].unique()

train_data = [([df.iat[i + j, 1] for j in range(5)], df.iat[i + 5, 1]) for i in range(115)]