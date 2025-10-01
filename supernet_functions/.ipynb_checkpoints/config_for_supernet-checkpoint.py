import numpy as np

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_ids' : [0]
    },
    'lookup_table' : {
        'create_from_scratch' : False,
        'path_to_lookup_table' : './supernet_functions/lookup_table.txt',
        'number_of_runs' : 50 # each operation run number_of_runs times and then we will take average
    },
    'logging' : {
        'path_to_log_file' : './logs/',
        'path_to_tensorboard_logs' : './logs/tensorboard'
    },
    'dataloading' : {
        'batch_size' : 128, # NSLKDD:128  UNSWNB15:128  CIC-IDS2017:512
        'w_share_in_train' : 1.0,
        'path_to_save_data' : './datasets'
        # 'path_to_save_data' : './datasets/NSL-KDD/KDDTrain+.txt'
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 1e-4,
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 1e-4,
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 0.2,
        'beta' : 0.6
    },
    'train_settings' : {
        'train_thetas_from_the_epoch' : 12, # 4
        'cnt_epochs' : 20, # 12
        'print_freq' : 50, # 50
        'sample_num' : 100, # 100
        'while_circle_count' : 30, # 用于防止死循环的计数器
        'path_to_save_model' : './models/',
        # for transformer block input/output dimension
        'transformer_block_dim': 128, # NSLKDD:128  UNSWNB15:128*2  CIC-IDS2017:128
        # for Gumbel Softmax
        'init_temperature' : 5.0,
        'exp_anneal_rate' : np.exp(-0.045),
        'class_num' : 5, # NSLKDD:5  UNSWNB15:10  CIC-IDS2017:15
    }
}