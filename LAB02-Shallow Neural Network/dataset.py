import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing



class MLDataset(Dataset):
    def __init__(self):
        # load data
        data = pd.read_csv('train.csv', encoding='utf-8')
        # label's columns name, no need to rewrite
        label_col = [
            'Input_A6_024' ,'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
            'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
            'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
            'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
        ]
        # ================================================================================ #
        # Do any operation on self.train you want with data type "dataframe"(recommanded) in this block.
        # For example, do normalization or dimension Reduction. 做標準化跟缺失值
        # Some of columns have "nan", need to drop row or fill with value first
        # For example:
        data = data.fillna(data.mean())  # 缺值補平均值
        # 標準化
        self.label = data[label_col]
        self.train = data.drop(label_col, axis=1)

        #data = data.fillna(0)              #缺值補0
        #缺失值補0的   Training loss: 0.0054 Training WRMSE: 0.3912

        #data = data.fillna(-1)             #缺值補-1
        #缺失值補-1的  Training loss: 0.0025 Training WRMSE: 0.2473
        # print(type(data))
        # data = data.fillna(data.mean())     #缺值補平均值
        #缺失值補均值的 Training loss: 0.0066 Training WRMSE: 0.4392
        normalization = preprocessing.MinMaxScaler( feature_range=(-1,1) )
        self.train = normalization.fit_transform(self.train)
        self.train = pd.DataFrame(self.train)


        #data = StandardScaler().fit_transform(data)

        # self.label = data[label_col]
        # self.train = data.drop(label_col, axis=1)

        # ================================================================================ #


    def __len__(self):
        #  no need to rewrite
        return len(self.train)

    def __getitem__(self, index):
        # transform dataframe to numpy array, no need to rewrite

        x = self.train.iloc[index, :].values
        y = self.label.iloc[index, :].values
        return x, y

