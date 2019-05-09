import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pandas import scatter_matrix

data_csv = '/home/pzhang/DataMining/Surface_vegetation/covtype.data'
new_data_csv = '/home/pzhang/DataMining/Surface_vegetation/covtype_categorical.csv'
data_small_csv = '/home/pzhang/DataMining/Surface_vegetation/covtype_categorical_small.csv'


def pre_process_data():
    # load raw data
    df = pd.read_csv(data_csv, header=None)
    print(df.shape)
    print(df.head())
    # Convert binary columns into categorical
    Wilderness_Area = df.iloc[:, 10:13 + 1].copy()
    Soil_Type = df.iloc[:, 14:53 + 1].copy()
    df = df.drop(np.arange(10, 13 + 1), axis=1)
    df = df.drop(np.arange(14, 53 + 1), axis=1)
    df.columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                  'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                  'Horizontal_Distance_To_Fire_Points', 'Cover_Type']

    df['Wilderness_Area'] = Wilderness_Area.idxmax(axis=1) - 10 + 1
    df['Soil_Type'] = Soil_Type.idxmax(axis=1) - 14 + 1
    print(df.shape)
    print(df.head())
    # df.to_csv(new_data_csv, index=False, header=True)
    # 检查是否有缺失值
    print(pd.DataFrame(df.isnull().sum()))
    # 每列最小值
    print(pd.DataFrame(df.min()))
    # 每列最大值
    print(pd.DataFrame(df.max()))

    pass


def change_datatype(df):
    print(df.info(memory_usage='deep'))
    df['Elevation'] = df['Elevation'].astype('uint16')
    df['Aspect'] = df['Aspect'].astype('uint16')
    df['Slope'] = df['Slope'].astype('uint8')
    df['Horizontal_Distance_To_Hydrology'] = df['Horizontal_Distance_To_Hydrology'].astype('uint16')
    df['Vertical_Distance_To_Hydrology'] = df['Vertical_Distance_To_Hydrology'].astype('int16')
    df['Horizontal_Distance_To_Roadways'] = df['Horizontal_Distance_To_Roadways'].astype('uint32')
    df['Hillshade_9am'] = df['Hillshade_9am'].astype('uint16')
    df['Hillshade_Noon'] = df['Hillshade_Noon'].astype('uint16')
    df['Hillshade_3pm'] = df['Hillshade_3pm'].astype('uint16')
    df['Horizontal_Distance_To_Fire_Points'] = df['Horizontal_Distance_To_Fire_Points'].astype('uint32')
    df['Cover_Type'] = df['Cover_Type'].astype('uint8')
    df['Wilderness_Area'] = df['Wilderness_Area'].astype('uint8')
    df['Soil_Type'] = df['Soil_Type'].astype('uint8')
    print(df.info(memory_usage='deep'))
    df.to_csv(data_small_csv, index=False, header=True)
    df = pd.read_csv(data_small_csv)
    print(df.shape)
    print(df.head())
    pass


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()


def column_plot(df):
    # df.hist(figsize=(20, 12))
    # plt.savefig('column_wise_plot.png')

    # df.plot(kind='box', figsize=(20, 6), rot=90, subplots=True)
    # plt.savefig('column_wise_box.png')

    # df.plot(kind='bar', figsize=(20, 6), rot=90, subplots=True)
    # plt.savefig('column_wise_bar.png')
    #
    # df.plot(kind='scatter', figsize=(20, 6), rot=90, subplots=True)
    # plt.savefig('column_wise_scatter.png')
    #
    # df.plot(kind='pie', figsize=(20, 6), rot=90, subplots=True)
    # plt.savefig('column_wise_pie.png')

    plot_confusion_matrix(df.corr().as_matrix(), classes=df.columns, title='Correlation')
    scatter_matrix(df, alpha=0.2, figsize=(20, 20))
    plt.savefig('scatter_matrix.png')

    df.plot(kind='scatter', x='Wilderness_Area', y='Soil_Type')
    plt.savefig('scatter.png')

    pass


def main():
    df = pd.read_csv(data_small_csv)
    column_plot(df)



    pass


if __name__ == '__main__':
    main()