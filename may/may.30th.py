import baostock as bs#用于连接 Baostock 金融数据 API。
import pandas as pd#用于处理和分析获取到的数据。
import matplotlib.pyplot as plt#用于绘制图表。
lg = bs.login()#发起登录请求，返回登录结果对象 lg。
if lg.error_code != '0':#状态码，'0' 表示成功，非零表示失败。
    print(f"登录失败：{lg.error_msg}")#lg.error_msg：失败时的错误信息（如网络问题、服务器繁忙）。
    exit()
rs = bs.query_history_k_data_plus(
    "600313.sh",## 股票代码：上交所A股后缀为.sh
    "date,open,high,low,close,volume",## 需要获取的字段
    start_date="2024-01-01",
    end_date="2025-01-01",
    frequency="d"## 数据频率：d=日线，w=周线，m=月线
)
data_list = []
while rs.error_code == '0' and rs.next():#rs.error_code：查询结果的状态码，'0' 表示成功。rs.next()：逐行遍历查询结果。
    data_list.append(rs.get_row_data())#rs.get_row_data()：获取当前行的数据，返回列表格式。
df = pd.DataFrame(data_list, columns=rs.fields)
print(df.describe())
bs.logout()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True) #将 date 列设置为索引，inplace=True 表示直接在原 DataFrame 上修改。
print(df.describe())
print(type(df.index),type(df))
print(df.index.dtype)
print(df['close'])
df['close'] = pd.to_numeric(df['close'], errors='coerce')#pd.to_numeric 函数会尝试将 close 列的数据转换为数值类型， errors='coerce' 参数表示如果遇到无法转换的值，会将其转换为 NaN 。
'''
df['close'].plot(figsize=(10, 6), label="600313.sh")
plt.title(f"600313.sh Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()## 显示图例（即 label 参数指定的内容）
plt.show()
'''
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()
df['Signal'] = 0
df.loc[df['SMA_50'] > df['SMA_200'], 'Signal'] = 1
df.loc[df['SMA_50'] < df['SMA_200'], 'Signal'] = -1
df['Daily_Return'] = df['close'].pct_change()#pct_change()：计算每天的收益率 = (今天收盘 - 昨天收盘) / 昨天收盘。
df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()#.cumprod()：累乘，计算总资产增长倍数。(1 + 每日收益)：每一天的增长因子。
strategy_performance = {
    'Total Return': df['Cumulative_Return'].iloc[-1] - 1,
    'Annualized Return': (df['Cumulative_Return'].iloc[-1] ** (252 / len(df))) - 1,
    'Max Drawdown': (df['Cumulative_Return'] / df['Cumulative_Return'].cummax() - 1).min(),#.cummax()历史最高点
}

print("策略表现:")
for key, value in strategy_performance.items():
    print(f"{key}: {value:.4f}")

# 绘制累计收益曲线
plt.figure(figsize=(10, 6))
plt.plot(df['Cumulative_Return'], label='Strategy Cumulative Return', color='b')
plt.plot(df['close'] / df['close'].iloc[0], label='Stock Cumulative Return', color='g')
plt.title("Cumulative Return of Strategy vs. Stock")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()


print(df)
