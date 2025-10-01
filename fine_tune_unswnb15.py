import os
import datetime
import numpy as np
import pandas as pd

import torch

from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from datetime import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_dis = torch.device('cuda:0')
# device_gen = torch.device('cuda:0')

# 获取当前日期和时间
run_time = datetime.now()
model_kind = f'UNSW-NB15_Multi'

save_path = os.path.join(f'sampled_model_results', model_kind)
save_path = os.path.join(save_path, str(run_time))

os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}', exist_ok=True)
os.makedirs(f'{save_path}/models', exist_ok=True)
os.makedirs(f'{save_path}/results', exist_ok=True)



# 基础参数设置
batch_size = 128
train_epoches = 30
class_num = 10
last_epoch = -1
print_interval = 100




def Load_UNSWNB15(path_train_data, path_test_data, batch_size, binary_or_multi='multi'):

    categorical_columns = ['proto', 'service', 'state']
    
    classification = ['Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS', 
                    'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
    # 加载数据 train_num:125973, test_num:22544, total_data:148517
    data_train = pd.read_csv(path_train_data).copy()
    data_test = pd.read_csv(path_test_data).copy()
    total_data = pd.concat([data_train, data_test], axis=0) # 合并train和test
    total_data = total_data.drop(['id'], axis=1)
    train_num = len(data_train)
    test_num = len(data_test)

    # 特征
    features = total_data.iloc[:, :-2]     
    
    # 标签（以Binary/Multi形式加载Y的值）
    if binary_or_multi=='binary':    
        # 删除attack_cat列
        total_data = total_data.drop('attack_cat', axis=1)
        # 把labels转换为binary[0,1] 
        labels = total_data.iloc[:, -1]
    elif binary_or_multi=='multi':
        # 删除label列
        total_data = total_data.drop('label', axis=1)
        labels_class = total_data.iloc[:, -1]
        
        pdlist_class_dict = {}
        for index, data_class in enumerate(classification):
            pdlist_class_dict[data_class] = index
                
        labels = np.array([pdlist_class_dict[row] for row in np.array(labels_class)])
        
    # One-hot编码数据
    features = pd.get_dummies(features, columns=categorical_columns)
    
    # Min-Max标准化
    scaler = MinMaxScaler().fit(features)
    features = scaler.transform(features)

    # 凑形状，增加60列
    addition_number = 60
    addition_data = np.zeros((len(total_data), addition_number))
    features = np.concatenate((features, addition_data), axis=1)
    
    # X_train = features[:train_num][:, :, np.newaxis]
    # X_test = features[train_num:][:, :, np.newaxis]
    X_train = features[:train_num].astype(np.float32)
    X_test = features[train_num:].astype(np.float32)
    Y_train = labels[:train_num].astype(np.longlong)
    Y_test = labels[train_num:].astype(np.longlong)
    
        
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # Y_train = torch.LongTensor(Y_train)
    # Y_test = torch.LongTensor(Y_test)

    # 创建tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader




# 加载数据
path_train_data='datasets/UNSW-NB15/UNSW_NB15_training-set.csv'
path_test_data='datasets/UNSW-NB15/UNSW_NB15_testing-set.csv'

train_loader, test_loader = Load_UNSWNB15(path_train_data=path_train_data, 
                                          path_test_data=path_test_data, 
                                          batch_size=batch_size,
                                          binary_or_multi='multi')# 装载数据到loader里面)




# 加载模型
sampled_path =  f''
dis = torch.load(sampled_path, weights_only=False)
dis = dis_ind.model.to(device_dis)

dis = nn.DataParallel(dis)




# 优化器、学习率
criterion = nn.CrossEntropyLoss()

params_dis = [param for param in dis.parameters()]
optimizer_dis = Adam(params=params_dis, lr=1e-4, weight_decay=1e-5)
scheduler_dis = CosineAnnealingLR(optimizer_dis, T_max=int(len(train_loader)/2), eta_min=5e-6, last_epoch=last_epoch)




# The main training code will be released once the article is published.