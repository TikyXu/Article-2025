import os
import datetime
import numpy as np

import torch
from torch import nn
# from torchsummary import summary
# # from torchinfo import summary
from tensorboardX import SummaryWriter

from general_functions.dataloaders import get_nslkdd_train_loader, get_nslkdd_test_loader
from general_functions.utils import get_logger, weights_init, create_directories_from_list
from supernet_functions.lookup_table_builder import LookUpTableTransformer
from supernet_functions.model_supernet import SuperNet_Generator, SuperNet_Discriminator
# from supernet_functions.model_supernet_backup import SuperNet_Generator, SuperNet_Discriminator
from supernet_functions.training_functions_supernet import TrainerSupernet
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device_dis = torch.device('cuda:0')
device_gen = torch.device('cuda:0')

def train_supernet():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True
    run_time = datetime.datetime.now()

    create_directories_from_list([CONFIG_SUPERNET['logging']['path_to_tensorboard_logs']])
    
    logger = get_logger(CONFIG_SUPERNET['logging']['path_to_log_file']+'AMEGAN_NSLKDD_Multi-Class '+str(run_time))
    writer = SummaryWriter(log_dir=CONFIG_SUPERNET['logging']['path_to_tensorboard_logs'])
    logger.info(f"AutoTRAN Training Start: {datetime.datetime.now()}")
    #### LookUp table consists all information about layers
    lookup_table = LookUpTableTransformer(calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'])
    
    #### DataLoading
    train_loader, _ = get_nslkdd_train_loader(train_portion=CONFIG_SUPERNET['dataloading']['w_share_in_train'],
                                              batch_size=CONFIG_SUPERNET['dataloading']['batch_size'],
                                              path_to_save_data=CONFIG_SUPERNET['dataloading']['path_to_save_data']+'/NSL-KDD/KDDTrain+.txt',
                                              binary_or_multi='multi')
    test_loader = get_nslkdd_test_loader(batch_size=CONFIG_SUPERNET['dataloading']['batch_size'],
                                         path_to_save_data=CONFIG_SUPERNET['dataloading']['path_to_save_data']+'/NSL-KDD/KDDTest+.txt',
                                         binary_or_multi='multi')
    
    
    # The main training code will be released once the article is published.
    
if __name__ == "__main__":
    
    train_supernet()