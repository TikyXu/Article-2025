import os
import datetime
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from datetime import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_dis = torch.device('cuda:0')
# device_gen = torch.device('cuda:0')

# 获取当前日期和时间
run_time = datetime.now()
model_kind = f'NSL-KDD_Multi'

save_path = os.path.join(f'sampled_model_results', model_kind)
save_path = os.path.join(save_path, str(run_time))

os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}', exist_ok=True)
os.makedirs(f'{save_path}/models', exist_ok=True)
os.makedirs(f'{save_path}/results', exist_ok=True)



# 基础参数设置
batch_size = 128
train_epoches = 60
class_num = 5
last_epoch = -1
print_interval = 200




def Load_NSLKDD(path_train_data, path_test_data, batch_size, binary_or_multi):
    # 列名，根据NSL-KDD数据集文档定义
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_hot_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "score"
        # "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]
    normal = ['normal']    
    dos    = ['back', 'land', 'neptune', 'pod', 'smurf', 
              'teardrop', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
    probe  = ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint']
    r2l    = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 
              'spy', 'warezclient', 'warezmaster', 'sendmail', 'named', 
              'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm']
    u2l    = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'httptunnel', 
              'ps', 'sqlattack', 'xterm']
    
    categorical_columns = ['protocol_type', 'service', 'flag']
        
    # 加载数据 train_num:125973, test_num:22544, total_data:148517
    data_train = pd.read_csv(path_train_data, header=None, names=column_names)
    data_test = pd.read_csv(path_test_data, header=None, names=column_names)
    total_data = pd.concat([data_train, data_test], axis=0) # 合并train和test
    train_num = len(data_train)
    # 删除Score列
    total_data = total_data.drop('score', axis=1)

    # 特征、标签
    features = total_data.iloc[:, :-1] 
    labels = total_data.iloc[:, -1]
    
    # One-hot编码数据
    features = pd.get_dummies(features, columns=categorical_columns)
    
    # Min-Max标准化
    scaler = MinMaxScaler().fit(features)
    features = scaler.transform(features)
    
    # 凑形状，增加6列
    addition_number = 6
    addition_data = np.zeros((len(total_data), addition_number))
    features = np.concatenate((features, addition_data), axis=1)

    pdlist_class_dict = {}
    for index, data_class in enumerate([normal, dos, probe, r2l, u2l]):
        for item in data_class:
            pdlist_class_dict[item] = index

    # 给表格数据赋值
    if binary_or_multi == 'multi':
        labels = np.array([pdlist_class_dict[row] for row in labels])
    elif binary_or_multi == 'binary':
        labels = np.array([0 if row=='normal' else 1 for row in labels])
    
    X_train = np.array(features[:train_num]).astype(np.float32)
    X_test = np.array(features[train_num:]).astype(np.float32)
    Y_train = np.array(labels[:train_num]).astype(np.longlong)
    Y_test = np.array(labels[train_num:]).astype(np.longlong)

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
path_train_data='datasets/NSL-KDD/KDDTrain+.txt'
path_test_data='datasets/NSL-KDD/KDDTest+.txt'
train_loader, test_loader = Load_NSLKDD(path_train_data=path_train_data, 
                                        path_test_data=path_test_data, 
                                        batch_size=batch_size,
                                        binary_or_multi='multi')




# 加载模型
sampled_path =  f''
dis_ind = torch.load(sampled_path, weights_only=False)
dis = dis_ind.model.to(device_dis)

dis = nn.DataParallel(dis)




# 优化器、学习率
criterion = nn.CrossEntropyLoss()

params_dis = [param for param in dis.parameters()]
optimizer_dis = Adam(params=params_dis, lr=1e-4, weight_decay=5e-4)
scheduler_dis = CosineAnnealingLR(optimizer_dis, T_max=int(len(train_loader)/2), eta_min=5e-6, last_epoch=last_epoch)




# The main training code will be released once the article is published.