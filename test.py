import pandas as pd

ForestCoverType_dir = '/mnt/hd/DataMining/Surface_vegetation/covtype.data'
weather_dir = '/mnt/hd/DataMining/weather/weather.data'

with open(weather_dir, 'r') as f:
    line = f.readlines()  # 调用文件的 readline()方法
    while line:
        print(line)
