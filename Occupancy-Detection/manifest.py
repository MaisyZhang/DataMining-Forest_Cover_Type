import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

train_txt = '/home/pzhang/DataMining/Indoor_living/datatraining.txt'
test_txt = '/home/pzhang/DataMining/Indoor_living/datatest.txt'

train_csv = '/home/pzhang/DataMining/Indoor_living/train.csv'
test_csv = '/home/pzhang/DataMining/Indoor_living/test.csv'


def visualize_data():
    train = pd.read_csv(train_txt)
    test = pd.read_csv(test_txt)

    print("Training Set")
    print(train.head())  # 几行的数据
    print(train.describe())  # 所有属性列的具体分析
    print(train.shape)  # (8143, 7)

    print("Test Set")
    print(test.head())
    print(test.describe())
    print(test.shape)

    pass


def sortout_data():
    '''
    数据清洗
    1. date --> Date
    2. 元数据的属性值和列对应不上，需要重新存储
    '''
    # for train
    lines = []
    with open(train_txt, 'r') as f:
        lines = f.readlines()

    new_lines = []
    new_lines.append(lines[0].replace('date', 'Date'))

    for line in lines[1:]:
        new_lines.append(','.join(l for l in line.split(',')[1:]))

    with open(train_csv, 'w') as f:
        f.writelines(new_lines)

    # for test
    lines = []
    with open(test_txt, 'r') as f:
        lines = f.readlines()

    new_lines = []
    new_lines.append(lines[0].replace('date', 'Date'))

    for line in lines[1:]:
        new_lines.append(','.join(l for l in line.split(',')[1:]))

    with open(test_csv, 'w') as f:
        f.writelines(new_lines)
    pass


def check_data(train, test):
    print(train.isnull().sum())
    print(test.isnull().sum())
    pass


def correlation_data(train):
    pd.scatter_matrix(train, c=train['Occupancy'], figsize=[10, 10])
    plt.savefig('occupancy_scatter_matrix.png')
    pass


def dateOrNotToDate(date_str):
    '''
    转换成py格式的时间
    '''
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def convert_dates(df):
    for i, date in enumerate(df['Date']):
        df.iloc[i, df.columns.get_loc('Date')] = dateOrNotToDate(date)


def time_series_feature(train, test):
    convert_dates(train)
    convert_dates(test)
    for i, col in enumerate(train.columns.values[1:]):
        plt.subplot(3, 2, i + 1)
        plt.plot(train['Date'].values.tolist(), train[col].values.tolist(), label=col)
        plt.title(col)
        fig, ax = plt.gcf(), plt.gca()
        ax.xaxis_date()
        fig.autofmt_xdate()
        fig.set_size_inches(10, 10)
        plt.tight_layout()
        plt.grid(True)

    plt.savefig('time_series_feature.png')
    pass


def analysize_occupancy():
    days = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]
    seventh_of_feb = datetime.strptime('2015-02-07', '%Y-%m-%d')
    print(days[seventh_of_feb.weekday()])
    pass


def gen_index_per_date(train):
    convert_dates(train)
    date_list = train.Date.values.tolist()
    day_start_indices = []
    for i in range(5, 11):
        day_start_indices.append(
            date_list.index(
                datetime.strptime(
                    '2015-02-' + str(i) + ' 00:00:00',
                    '%Y-%m-%d %H:%M:%S'
                )
            )
        )
    day_start_indices = [0] + day_start_indices
    print(day_start_indices)

    for i in range(len(day_start_indices)):
        plt.subplot(4, 2, i + 1)
        if i != len(day_start_indices) - 1:
            plt.plot(
                date_list[day_start_indices[i]:day_start_indices[i + 1]],
                train['Occupancy'].values.tolist()[
                day_start_indices[i]:day_start_indices[i + 1]])
        else:
            plt.plot(
                date_list[day_start_indices[i]:],
                train['Occupancy'].values.tolist()[day_start_indices[i]:])
        plt.title(str(i + 4) + 'th of Feb.')
        plt.grid(True)
        plt.xticks(rotation=90)
        fig, ax = plt.gcf(), plt.gca()
        ax.xaxis_date()
        fig.set_size_inches(10, 10)
        fig.tight_layout()

    plt.savefig('time_series_occupancy.png')
    pass


def print_per_day_occupancy(train):
    convert_dates(train)
    date_list = train.Date.values.tolist()
    day_start_indices = []
    for i in range(5, 11):
        day_start_indices.append(
            date_list.index(
                datetime.strptime(
                    '2015-02-' + str(i) + ' 00:00:00',
                    '%Y-%m-%d %H:%M:%S'
                )
            )
        )
    day_start_indices = [0] + day_start_indices
    print(day_start_indices)
    print('Daily Work Hours')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    for i in range(len(day_start_indices) - 1):
        try:
            print('Start:\t',
                  train.loc[(train.Date > date_list[day_start_indices[i]]) &
                            (train.Date < date_list[day_start_indices[i + 1]]) &
                            (train.Occupancy == 1), 'Date'].iloc[0])
            print('End:\t',
                  train.loc[(train.Date > date_list[day_start_indices[i]]) &
                            (train.Date < date_list[day_start_indices[i + 1]]) &
                            (train.Occupancy == 1), 'Date'].iloc[-1])
        except:
            print('No Occupancy')
        print('########################################')
        print()
    pass


def analysize_light(train):
    convert_dates(train)
    date_list = train.Date.values.tolist()
    day_start_indices = []
    for i in range(5, 11):
        day_start_indices.append(
            date_list.index(
                datetime.strptime(
                    '2015-02-' + str(i) + ' 00:00:00',
                    '%Y-%m-%d %H:%M:%S'
                )
            )
        )
    day_start_indices = [0] + day_start_indices
    lighting = train.loc[
        (train.Date > date_list[day_start_indices[3]]) &
        (train.Date < date_list[day_start_indices[4]]) &
        (train.Light > 850),
        ('Date', 'Light')
    ]
    print(lighting)
    pass


def gen_new_dataset(train, test):
    days = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]
    convert_dates(train)
    convert_dates(test)
    def add_features(df):
        df.loc[:, 'Weekend'] = 0
        df.loc[:, 'WorkingHour'] = 0

        for i, date in enumerate(df['Date']):
            if (days[date.weekday()] == 'Saturday') or \
                    (days[date.weekday()] == 'Sunday'):
                df.iloc[i, df.columns.get_loc('Weekend')] = 1

            if date.time() >= datetime.strptime('07:30', '%H:%M').time() and \
                    date.time() <= datetime.strptime('18:00', '%H:%M').time():
                df.iloc[i, df.columns.get_loc('WorkingHour')] = 1

    add_features(train)
    add_features(test)

    pd.scatter_matrix(train, c=train['Occupancy'], figsize=[15, 15])
    plt.savefig('new_occupancy_scatter_matrix.png')
    pass


def main():

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # ① 统计数据
    # visualize_data()
    # ② 整理数据格式
    # sortout_data()
    # ③ 检查NaN数据 and
    # check_data(train, test)
    # ④ 可视化数据属性之间的相关性
    # correlation_data(train)
    # ⑤ 转换时间，想查看时间维度各个特征的关系
    # time_series_feature(train, test)
    # ⑥ 根据⑤中结果，继续分析occupancy，是不是weekend，那么算一下
    # analysize_occupancy()
    # ⑦ 从5号到10号，得到每一天的索引， 并画出每一天的占有率
    # gen_index_per_date(train)
    # ⑧ 打印每天的入住率
    # print_per_day_occupancy(train)  # officers do not come to office before 07:30 and they depart after 18:00.
    # ⑨ 分析属性：Light， 由⑤中可以看出有异常值，那么输出一下 # May be a lightning?
    # analysize_light(train)
    # ⑩ 分析属性：CO2，可以直接通过⑤中的图看出
    # ①① 经过以上的分析，讲weekend和workinghours添加为功能，重新修改数据集
    # 对于周末，正如所料，我会检查日期是“周六还是周日”。 如果是，则Weekend = 1，否则Weekend = 0
    # 对于WorkingHours，如果一天的时间在07:30到18:00之间，那么WorkingHours = 1，否则WorkingHours = 0
    # gen_new_dataset(train, test)



    pass


if __name__ == '__main__':
    main()