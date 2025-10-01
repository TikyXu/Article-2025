import os
import csv
import copy
import math
import time
import random
import datetime
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from scipy.special import softmax
from torch.autograd import Variable

from general_functions.utils import AverageMeter, save, accuracy, check_tensor_in_list, writh_new_ARCH_to_fbnet_transformer_modeldef
from general_functions.nsga import Individual, Population, IndividualQueue
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.lookup_table_builder import CANDIDATE_BLOCKS_TRANSFORMER, SEARCH_SPACE_TRANSFORMER
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning, message="RNN module weights are not part of single contiguous chunk of memory")

class TrainerSupernet:
    
    def __init__(self, device_dis, device_gen, logger, writer, run_time, lookup_table, class_number_count):

        self.device_dis = device_dis
        self.device_gen = device_gen       
        self.logger       = logger
        self.writer       = writer
        self.run_time     = run_time
        self.lookup_table = lookup_table
        
        self.while_circle_count = CONFIG_SUPERNET['train_settings']['while_circle_count'] # 用于防止死循环的计数器
        self.sample_number = CONFIG_SUPERNET['train_settings']['sample_num'] # 每一迭代的种群数量
        self.sample_number_dis = 0         # 计数Discriminator下一迭代TopK+Random采样的个体数量
        self.sample_number_gen = 0         # 计数Generator下一迭代TopK+Random采样的个体数量
        self.last_iteration_code_dis = []  # 上一次迭代的非支配Discriminator编码，及其交叉、变异编码
        self.last_iteration_code_gen = []  # 上一次迭代的非支配Generator编码，及其交叉、变异编码
        self.overall_nondominated_dis = [] # 全局Discriminator非支配解集
        self.overall_nondominated_gen = [] # 全局Generator非支配解集
        
        self.best_discriminator_supernet = None # Discriminator的Supernet
        self.best_discriminator = None          # Discriminator的最优子网络
        self.best_generator_supernet = None     # Generator的Supernet
        self.best_generator = None              # Generator的最优子网络
        
        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model']
        self.class_num                   = CONFIG_SUPERNET['train_settings']['class_num']

        self.save_model_path = os.path.join(self.path_to_save_model, str(self.run_time)) # 模型保存路径
        self.class_number_count          = class_number_count # 数据集每个类别的样本数量
        # 计算数据集每个类别样本数量的反比，Inverse Proportional Ratio
        class_ratio = [item/sum(self.class_number_count) for item in self.class_number_count]
        minus_log = [-math.log(item) for item in class_ratio]
        self.invert_ratio = [log/sum(minus_log) for log in minus_log]
        self.class_index = None


    def train_loop(self, train_loader, test_loader, generator_supernet, discriminator_supernet):

        self.class_index = [(train_loader.dataset.label==i).nonzero(as_tuple=True)[0].tolist() for i in range(self.class_num)]

        """
        Supernet预训练: 训练模型weights
        """
        self.logger.info(f"\nSupernet pretraining from epochs 1 ~ {self.train_thetas_from_the_epoch}\n")
        for epoch in range(self.train_thetas_from_the_epoch):
           
            self.logger.info(f"Weights training -- Epoch{epoch+1}\n")
            self._training_step(weight_or_theta='weight',
                                generator_supernet=generator_supernet, 
                                discriminator_supernet=discriminator_supernet, 
                                loader=train_loader,
                                test_loader=test_loader,
                                epoch=epoch,)
            
            generator_supernet = self.best_generator_supernet
            discriminator_supernet = self.best_discriminator_supernet
            
            self.logger.info('')

        """
        架构参数搜索: 训练结构参数theta
        """
        self.logger.info(f"Architecture searching from epochs {self.train_thetas_from_the_epoch+1} ~ {self.cnt_epochs}\n")
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            
            self.logger.info(f"Weights & Theta training -- Epoch{epoch+1}\n")
            self._training_step(weight_or_theta='theta',
                                generator_supernet=generator_supernet, 
                                discriminator_supernet=discriminator_supernet, 
                                loader=train_loader,
                                test_loader=test_loader,
                                epoch=epoch,)
            
            generator_supernet = self.best_generator_supernet
            discriminator_supernet = self.best_discriminator_supernet
            
            self.temperature = self.temperature * self.exp_anneal_rate
            self.logger.info('')
            
            self.logger.info(f"Model Epoch {epoch+1}")
            for dis_individual in self.overall_nondominated_dis:
                self.logger.info(f'Discriminator,Epo:{dis_individual.epoch+1},Iter:{dis_individual.iteration+1},Code:{dis_individual.code},Obj:{dis_individual.objectives}')
                if dis_individual.epoch == epoch: # 保存当前迭代产生的非支配解
                    self._model_save(model=dis_individual, gen_or_dis='Discriminator',
                                    epoch=dis_individual.epoch, iteration=dis_individual.iteration, 
                                    code=dis_individual.code, objectives=dis_individual.objectives, 
                                    save_path=self.save_model_path)
                    
            for gen_individual in self.overall_nondominated_gen:
                self.logger.info(f'Generator,Epo:{gen_individual.epoch},Iter:{gen_individual.iteration},Code:{gen_individual.code},Obj:{gen_individual.objectives}')   
                if dis_individual.epoch == epoch: # 保存当前迭代产生的非支配解
                    self._model_save(model=gen_individual, gen_or_dis='Generator',
                                    epoch=gen_individual.epoch, iteration=gen_individual.iteration, 
                                    code=gen_individual.code, objectives=gen_individual.objectives, 
                                    save_path=self.save_model_path)
            self.logger.info('')

        self.logger.info(f'Sampling Final Model')
        """
        采样并筛选最终Discriminator
        """
        # 保存Discriminator Supernet
        save_path = self.save_model_path+'/Discriminator/Supernet/'
        os.makedirs(save_path, exist_ok=True) # Create directories if needed, ignore if already exist
                    
        torch.save(discriminator_supernet, save_path+'Final_Discriminator_Supernet.pth')

        final_dis_population = Population()
        # Max采样
        final_max_discriminator, final_max_discriminator_latency, final_max_discriminator_code = self._sample(discriminator_supernet, mode='Max')
        start_time_dis = time.time()
        final_max_discriminator.eval()
        with torch.no_grad():
            output_vaild = torch.empty(0).cuda(non_blocking=True)
            labels_vaild = torch.empty(0).cuda(non_blocking=True)
            for test_image, test_label in test_loader:                            
                test_image, test_label = test_image.view(test_image.shape[0], -1, test_image.shape[1]).cuda(non_blocking=True), test_label.cuda(non_blocking=True)
                out, _ = final_max_discriminator(test_image, self.temperature, 0, False)
                output_vaild = torch.cat((output_vaild, out), dim=0)
                label_one_hot = nn.functional.one_hot(test_label.type(torch.int64), num_classes=self.class_num).to(torch.float32).view(test_image.shape[0], 1, -1).cuda(non_blocking=True)
                labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
            
            final_max_accu = self._accuracy(output_vaild, labels_vaild)
            final_dis_population.add(Individual(epoch=self.cnt_epochs, iteration=0, 
                                                model=final_max_discriminator, code=final_max_discriminator_code,
                                                objectives=[pow(final_max_discriminator_latency, -1), final_max_accu]))
        end_time_dis = time.time()
        self.logger.info(f'Discriminator Code:{final_max_discriminator_code}; Accu:{final_max_accu:.4f}; Lat:{pow(final_max_discriminator_latency, -1):.4f}; Time:{end_time_dis-start_time_dis}s')

        # TopK采样
        for _ in range(int(1/4*self.sample_number)-1):
            # 采样TopK个体,且规避重复
            while True:
                final_topk_discriminator, final_topk_discriminator_latency, final_topk_discriminator_code = self._sample(discriminator_supernet, mode='TopK')
                if final_topk_discriminator_code not in final_dis_population.codes:
                    break

            # 计算Accuracy
            start_time_dis = time.time()
            final_topk_discriminator.eval()
            with torch.no_grad():
                output_vaild = torch.empty(0).cuda(non_blocking=True)
                labels_vaild = torch.empty(0).cuda(non_blocking=True)
                for test_image, test_label in test_loader:                            
                    test_image, test_label = test_image.view(test_image.shape[0], -1, test_image.shape[1]).cuda(non_blocking=True), test_label.cuda(non_blocking=True)
                    out, _ = final_topk_discriminator(test_image, self.temperature, 0, False)
                    output_vaild = torch.cat((output_vaild, out), dim=0)
                    label_one_hot = nn.functional.one_hot(test_label.type(torch.int64), num_classes=self.class_num).to(torch.float32).view(test_image.shape[0], 1, -1).cuda(non_blocking=True)
                    labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                
                final_topk_accu = self._accuracy(output_vaild, labels_vaild)

            final_dis_population.add(Individual(epoch=self.cnt_epochs, iteration=0,
                                                model=final_topk_discriminator, code=final_topk_discriminator_code,
                                                objectives=[pow(final_topk_discriminator_latency, -1), final_topk_accu]))
        
            end_time_dis = time.time()
            self.logger.info(f'Discriminator Code:{final_topk_discriminator_code}; Accu:{final_topk_accu:.4f}; Lat:{pow(final_topk_discriminator_latency, -1):.4f}; Time:{end_time_dis-start_time_dis}s')
        
        for dis in final_dis_population.fronts:
            self._model_save(model=dis, gen_or_dis='Discriminator',
                             epoch=dis.epoch, iteration=dis.iteration, 
                             code=dis.code, objectives=dis.objectives, 
                             save_path=self.save_model_path)

        """
        采样并筛选最终Generator
        """
        # 保存Generator Supernet
        save_path = self.save_model_path+'/Generator/Supernet/'
        os.makedirs(save_path, exist_ok=True) # Create directories if needed, ignore if already exist
                    
        torch.save(generator_supernet, save_path+'Final_Generator_Supernet.pth')

        fake_label_gen = np.array([self._weighted_random_int() for _ in range(CONFIG_SUPERNET['dataloading']['batch_size'])])
        validate_label = fake_label_gen.copy()
        fake_label_gen = torch.from_numpy(fake_label_gen).to(int).cuda(non_blocking=True)
        fake_label_one_hot_gen = nn.functional.one_hot(fake_label_gen, num_classes=CONFIG_SUPERNET['train_settings']['class_num']).to(torch.float32).view(CONFIG_SUPERNET['dataloading']['batch_size'], 1, -1).cuda(non_blocking=True)
        noise = torch.tensor(np.random.normal(0, 1, (CONFIG_SUPERNET['dataloading']['batch_size'], 1, 100)), dtype=torch.float32, requires_grad=False).cuda(non_blocking=True) 

        validate_image = self._get_data_from_label(train_loader, self.class_number_count, fake_label_gen)
        validate_image = validate_image.view(validate_image.shape[0], -1, validate_image.shape[-1]).cuda(non_blocking=True)
        validate_label = torch.from_numpy(validate_label).to(torch.float32).cuda(non_blocking=True)

        final_gen_population = Population()

        # Max采样
        final_max_generator, _, final_max_generator_code = self._sample(generator_supernet, mode='Max')        
        
        start_time_gen = time.time()
        with torch.no_grad():
            max_sampled_gen_images = final_max_generator(noise, fake_label_one_hot_gen, self.temperature, 0, False)

        max_sampled_gen_images = max_sampled_gen_images.view(CONFIG_SUPERNET['dataloading']['batch_size'], 1, -1).cuda(non_blocking=True)
            
        max_sampled_gen_images_output, max_sampled_pe_code_fake = self.best_discriminator(max_sampled_gen_images, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
        # Accuracy
        generator_max_sampled_model_accuracy = self._accuracy(output=max_sampled_gen_images_output, target=fake_label_gen)
        # Diversity
        max_sampled_inception_score = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(max_sampled_gen_images_output, dim=1), fake_label_one_hot_gen)
        _, max_sampled_pe_code_validate = self.best_discriminator(validate_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
        generator_max_sampled_model_diversity = self._diversity(inception_score=max_sampled_inception_score.cuda(non_blocking=True), 
                                                                pe_code_fake=max_sampled_pe_code_fake, 
                                                                pe_code_validate=max_sampled_pe_code_validate, 
                                                                validate_label=validate_label)                     
        
        final_gen_population.add(Individual(epoch=self.cnt_epochs, iteration=0, 
                                            model=final_max_generator, code=final_max_generator_code,
                                            objectives=[generator_max_sampled_model_diversity, generator_max_sampled_model_accuracy]))
        end_time_gen = time.time()
        self.logger.info(f'Generator Code:{final_max_generator_code}; Accu:{generator_max_sampled_model_accuracy:.4f}; Div:{generator_max_sampled_model_diversity:.4f}; Time:{end_time_gen-start_time_gen}s')

        # TopK采样
        for _ in range(int(1/4*self.sample_number)-1):
            # 采样TopK个体,且规避重复
            while True:
                final_topk_generator, _, final_topk_generator_code = self._sample(generator_supernet, mode='TopK')
                if final_topk_generator_code not in final_gen_population.codes:
                    break

            # 计算Accuracy、Diversity
            start_time_gen = time.time()
            final_topk_generator.eval()
            with torch.no_grad():
                final_topk_generator_images = final_topk_generator(noise, fake_label_one_hot_gen, self.temperature, 0, False)

            final_topk_generator_images = final_topk_generator_images.view(CONFIG_SUPERNET['dataloading']['batch_size'], 1, -1).cuda(non_blocking=True)
            
            final_topk_generator_images_output, final_topk_generator_pe_code_fake = self.best_discriminator(final_topk_generator_images, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
            # Accuracy
            final_topk_generator_accuracy = self._accuracy(output=final_topk_generator_images_output, target=fake_label_gen)
            # Diversity
            final_topk_generator_inception_score = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(final_topk_generator_images_output, dim=1), fake_label_one_hot_gen)
            _, final_topk_generator_pe_code_validate = self.best_discriminator(validate_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
            final_topk_generator_diversity = self._diversity(inception_score=final_topk_generator_inception_score.cuda(non_blocking=True), 
                                                             pe_code_fake=final_topk_generator_pe_code_fake, 
                                                             pe_code_validate=final_topk_generator_pe_code_validate, 
                                                             validate_label=validate_label)                     
            
            final_gen_population.add(Individual(epoch=self.cnt_epochs, iteration=0, 
                                                model=final_topk_generator, code=final_topk_generator_code,
                                                objectives=[final_topk_generator_diversity, final_topk_generator_accuracy]))
            end_time_gen = time.time()
            self.logger.info(f'Generator Code:{final_topk_generator_code}; Accu:{final_topk_generator_accuracy:.4f}; Div:{final_topk_generator_diversity:.4f}; Time:{end_time_gen-start_time_gen}s')

        for gen in final_gen_population.fronts:
            self._model_save(model=gen, gen_or_dis='Generator',
                             epoch=gen.epoch, iteration=gen.iteration, 
                             code=gen.code, objectives=gen.objectives, 
                             save_path=self.save_model_path)
            
    def _training_step(self, weight_or_theta, generator_supernet, discriminator_supernet, loader, test_loader, epoch):
        iteration = 0
        last_epoch = -1

        # generator_supernet = generator_supernet.train()
        # discriminator_supernet = discriminator_supernet.train()

        # 生成一个随机的 random_state
        random_state = np.random.seed(1337)
        # print(f'RANDOM STATE -- {random_state}')
        # 构造验证集
        test_image, test_label = test_loader.dataset[:]
        _, validate_image, _, validate_label = train_test_split(test_image, test_label, test_size=0.01, random_state=random_state)
        
        validate_image = validate_image.view(validate_image.shape[0], 1, -1).cuda(non_blocking=True)
        validate_label = validate_label.type(torch.float32).cuda(non_blocking=True)

        # 初始化Generator
        if self.best_generator_supernet == None:
            self.best_generator_supernet = generator_supernet
        if self.best_generator == None:
            self.best_generator, _, _ = self._sample(generator_supernet, mode='Random')

        # Discriminator 优化器、学习率
        # 训练超网权重时
        thetas_params_dis = [param for name, param in discriminator_supernet.named_parameters() if 'thetas' in name]
        params_except_thetas_dis = [param for param in discriminator_supernet.parameters() if not check_tensor_in_list(param, thetas_params_dis)]

        weight_optimizer_dis = torch.optim.Adam(params=params_except_thetas_dis, 
                                            lr=CONFIG_SUPERNET['optimizer']['w_lr'],
                                            weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
        
        weight_scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer_dis,
                                                                        T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                                        last_epoch=last_epoch)
        # 训练超网结构参数时
        if weight_or_theta=='theta':                
                
            layer_wise_thetas_dis = [param for name, param in discriminator_supernet.named_parameters() if 'thetas' in name]
            layer_wise_theta_optimizers_dis = [torch.optim.Adam(params=[layer], lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                                                weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
                                                                for layer in layer_wise_thetas_dis]
            # Automatically optimizes learning rate
            layer_wise_theta_schedulers_dis = [torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1, 
                                                                                        last_epoch=last_epoch)
                                                                                        for optimizer in layer_wise_theta_optimizers_dis]
        # Generator优化器、学习率
        # 训练超网权重时
        thetas_params_gen = [param for name, param in generator_supernet.named_parameters() if 'thetas' in name]
        params_except_thetas_gen = [param for param in generator_supernet.parameters() if not check_tensor_in_list(param, thetas_params_gen)]

        weight_optimizer_gen = torch.optim.Adam(params=params_except_thetas_gen, 
                                            lr=CONFIG_SUPERNET['optimizer']['w_lr'], 
                                            weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
        # Automatically optimizes learning rate
        weight_scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer_gen, 
                                                                        T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'], 
                                                                        last_epoch=last_epoch)
        # 训练超网结构参数时
        if weight_or_theta=='theta':  
            
            layer_wise_thetas_gen = [param for name, param in generator_supernet.named_parameters() if 'thetas' in name]
            layer_wise_theta_optimizers_gen = [torch.optim.Adam(params=[layer], lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                                                weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
                                                                for layer in layer_wise_thetas_gen]
            # Automatically optimizes learning rate
            layer_wise_theta_schedulers_gen = [torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1, 
                                                                                        last_epoch=last_epoch)
                                                                                        for optimizer in layer_wise_theta_optimizers_gen]


        for image, label in loader:
            batch_num = len(loader)
            batch_size = image.shape[0]
            
            image, label = image.view(batch_size, -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
            real_label_one_hot = nn.functional.one_hot(label.to(torch.int64), num_classes=self.class_num).to(torch.float32).view(batch_size, 1, -1).cuda(non_blocking=True)

            # 按照label标签构建生成样本，使得判别器输入的真假样本配对
            fake_label = label.clone().to(torch.int64)
            fake_label_one_hot_dis = nn.functional.one_hot(fake_label, num_classes=self.class_num).to(torch.float32).view(batch_size, 1, -1).cuda(non_blocking=True)
            
            noise = torch.tensor(np.random.normal(0, 1, (batch_size, 1, 100)), dtype=torch.float32, requires_grad=False).cuda(non_blocking=True) 
            latency_to_accumulate_dis = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True)
            latency_to_accumulate_gen = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True)
            
            # # 训练theta的时候采用sample的generator
            # if weight_or_theta=='theta':
            #     self.best_generator.eval()
            #     with torch.no_grad():
            #         fake_image_dis = self.best_generator(noise, fake_label_one_hot_dis, self.temperature, latency_to_accumulate_gen, False)                
            # # 训练w的时候采用Supernet的generator
            # else:
            self.best_generator_supernet.eval()
            with torch.no_grad():
                fake_image_dis, _ = self.best_generator_supernet(noise, fake_label_one_hot_dis, self.temperature, latency_to_accumulate_gen, True)
            
            fake_image_dis = fake_image_dis.view(batch_size, 1, -1).cuda(non_blocking=True)
            
            
            """""""""""""""
            Discriminator训练
            """""""""""""""            
            # # Discriminator 优化器、学习率
            # thetas_params_dis = [param for name, param in discriminator_supernet.named_parameters() if 'thetas' in name]
            # params_except_thetas_dis = [param for param in discriminator_supernet.parameters() if not check_tensor_in_list(param, thetas_params_dis)]

            # weight_optimizer_dis = torch.optim.Adam(params=params_except_thetas_dis, 
            #                                    lr=CONFIG_SUPERNET['optimizer']['w_lr'],
            #                                    weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
            # # 自动调整学习率
            # weight_scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer_dis,
            #                                                              T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
            #                                                              last_epoch=last_epoch)
            
            
            """
            Discriminator Supernet 训练 ———— 训练weights
            """
            discriminator_supernet.train()
            
            weight_optimizer_dis.zero_grad() # 梯度归零

            image.requires_grad_(True)
            fake_image_dis = fake_image_dis.detach()
            fake_image_dis = fake_image_dis.requires_grad_(True)
            # 真实/生成数据的原始logits，未经softmax处理
            outs_real_dis, _ = discriminator_supernet(image, self.temperature, latency_to_accumulate_dis, True)
            outs_fake_dis, _ = discriminator_supernet(fake_image_dis, self.temperature, latency_to_accumulate_dis, True)
            
            mean_real_dis = outs_real_dis.mean(dim=0, keepdim=True)
            mean_fake_dis = outs_fake_dis.mean(dim=0, keepdim=True)

            # 对于真实样本，我们希望 D(real) 相对于 mean_D_fake 在正确类别上的logit更高
            # 判别器希望 D_real 减去 mean_D_fake 后的结果，其对应真实标签的logit最高
            # 这里的目标标签是真实图片的类别标签 real_labels
            real_loss_dis = nn.CrossEntropyLoss()(outs_real_dis-mean_fake_dis, label.to(torch.int64))
            # 对于虚假样本，我们希望 D(fake) 相对于 mean_D_real 在“虚假”类别上的logit更高
            # 判别器希望 D_fake 减去 mean_D_real 后的结果，其对应“虚假”类别的logit最高
            # “虚假”类别的索引通常是 num_real_classes
            fake_loss_dis = nn.CrossEntropyLoss()(outs_fake_dis-mean_real_dis, fake_label)

            # R1+R2
            r1_penalty = self._zero_centered_gradient_penalty(image, outs_real_dis)
            r2_penalty = self._zero_centered_gradient_penalty(fake_image_dis, outs_fake_dis)
            
            dis_adversarial_loss = real_loss_dis + fake_loss_dis


            gamma = self._cosine_decay_with_warmup(cur_nimg=(iteration+1)*batch_size, base_value=2.,
                                                   total_nimg=(batch_num)*batch_size, final_value=0.2,
                                                   warmup_value=0.0, warmup_nimg=0, hold_base_value_nimg=0)
            discriminatorLoss = dis_adversarial_loss + (gamma / 2) * (r1_penalty + r2_penalty)
            
            dis_loss = discriminatorLoss.mean()
            
            dis_loss.backward()
            weight_optimizer_dis.step() # 更新模型参数
            weight_scheduler_dis.step() # 更新学习率

            self.best_discriminator_supernet = discriminator_supernet                   
            
            # 计算supernet的accuracy
            pred_multi_class = torch.cat((outs_real_dis, outs_fake_dis), dim=0)
            pred = torch.argmax(pred_multi_class, dim=1)
            target = torch.cat((label.to(torch.int64), fake_label), dim=0)

            dis_accuracy, precision, recall, f1_score = self._evaluation(model_kind='Generator',
                                                                         num_class=self.class_num,
                                                                         pred=pred, 
                                                                         target=target)
            
            self.logger.info(f'Dis weights Iter:{iteration+1}, Accu:{dis_accuracy:.4f}, Loss:{dis_loss:.4f}')

            """
            Discriminator子网采样、评估、选择 ———— 训练thetas
            """
            population_dis = Population()
            nondominated_queue_dis = IndividualQueue()
            cross_and_mutate_queue_dis = IndividualQueue()

            if weight_or_theta=='theta':                
                
                # layer_wise_thetas_dis = [param for name, param in discriminator_supernet.named_parameters() if 'thetas' in name]
                # layer_wise_theta_optimizers_dis = [torch.optim.Adam(params=[layer], lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                #                                                     weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
                #                                                     for layer in layer_wise_thetas_dis]
                # # Automatically optimizes learning rate
                # layer_wise_theta_schedulers_dis = [torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1, 
                #                                                                           last_epoch=last_epoch)
                #                                                                           for optimizer in layer_wise_theta_optimizers_dis]
                                
                for opt in layer_wise_theta_optimizers_dis:
                    opt.zero_grad()

                """
                采样、评估、筛选子网络
                """
                # 开始权重训练的第1个Epoch只有Random采样
                if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                # if epoch == 0 and iteration == 0:
                    for i in range(self.sample_number):
                        # 采样Random个体,且规避重复
                        while True:
                            discriminator_random_sampled_model, discriminator_random_sampled_latency, discriminator_random_sampled_code = self._sample(discriminator_supernet, mode='Random')
                            if discriminator_random_sampled_code not in population_dis.codes:
                                break
                        # print(f'Dis Code[{i+1}]:{discriminator_random_sampled_code}')
                        # 计算Accuracy
                        discriminator_random_sampled_model.eval()
                        with torch.no_grad():
                            val_output, _ = discriminator_random_sampled_model(validate_image, self.temperature, 0, False)
                        discriminator_random_sampled_model_accuracy = self._accuracy(output=val_output, target=validate_label)

                        population_dis.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=discriminator_random_sampled_model, code=discriminator_random_sampled_code,
                                                      objectives=[pow(discriminator_random_sampled_latency, -1), discriminator_random_sampled_model_accuracy]))
                       
                else:
                    # 上一迭代非支配解集及其交叉、变异个体
                    for individual_dis_code in self.last_iteration_code_dis:

                        discriminator_survival_model, discriminator_survival_latency = self._sample_from_code(discriminator_supernet, individual_dis_code)
                
                        # 计算Accuracy
                        discriminator_survival_model.eval()
                        with torch.no_grad():
                            val_output, _ = discriminator_survival_model(validate_image, self.temperature, 0, False)
                        discriminator_survival_model_accuracy = self._accuracy(output=val_output, target=validate_label)

                        population_dis.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=discriminator_survival_model, code=individual_dis_code,
                                                      objectives=[pow(discriminator_survival_latency, -1), discriminator_survival_model_accuracy]))
                    
                    # TopK采样
                    for _ in range(int(1/4*self.sample_number)):
                        # 采样TopK个体,且规避重复
                        while True:
                            discriminator_topk_sampled_model, discriminator_topk_sampled_latency, discriminator_topk_sampled_code = self._sample(discriminator_supernet, mode='TopK')
                            if discriminator_topk_sampled_code not in population_dis.codes:
                                break

                        # 计算Accuracy
                        discriminator_topk_sampled_model.eval()
                        with torch.no_grad():
                            val_output, _ = discriminator_topk_sampled_model(validate_image, self.temperature, 0, False)
                        discriminator_topk_sampled_model_accuracy = self._accuracy(output=val_output, target=validate_label)

                        population_dis.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=discriminator_topk_sampled_model, code=discriminator_topk_sampled_code,
                                                      objectives=[pow(discriminator_topk_sampled_latency, -1), discriminator_topk_sampled_model_accuracy]))
                        
                    # Random采样
                    for _ in range(self.sample_number_dis):                        
                        # 采样TopK个体,且规避重复
                        while True:
                            discriminator_random_sampled_model, discriminator_random_sampled_latency, discriminator_random_sampled_code = self._sample(discriminator_supernet, mode='Random')
                            if discriminator_random_sampled_code not in population_dis.codes:
                                break

                        # 计算Accuracy
                        discriminator_random_sampled_model.eval()
                        with torch.no_grad():
                            val_output, _ = discriminator_random_sampled_model(validate_image, self.temperature, 0, False)
                        discriminator_random_sampled_model_accuracy = self._accuracy(output=val_output, target=validate_label)
                        
                        population_dis.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=discriminator_random_sampled_model, code=discriminator_random_sampled_code,
                                                      objectives=[pow(discriminator_random_sampled_latency, -1), discriminator_random_sampled_model_accuracy]))
                
                # 保存非支配解集model,并构建非支配队列
                for individual in population_dis.fronts:
                    nondominated_queue_dis.insert(individual.code)
                    # self._model_save(model=individual.model, 
                    #                  gen_or_dis='Discriminator', epoch=individual.epoch, 
                    #                  iteration=individual.iteration, code=individual.code, 
                    #                  objectives=individual.objectives, save_path=self.save_model_path)
                
                self.best_discriminator = population_dis.fronts[0].model
                # 计算全局最优非支配解，保存至self.overall_nondominated_dis
                if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                    self.overall_nondominated_dis = population_dis.fronts
                else:
                    overall_population_dis = Population()  # 构建种群，便于计算全局非支配解集
                    all_nondominated_dis = self.overall_nondominated_dis # 初始化全部非支配解集all_nondominated_dis，并将前一迭代的非支配解集添加进去
                    for indiv in population_dis.fronts: # 往all_nondominated_dis添加当前迭代的非支配解集
                        all_nondominated_dis.append(indiv)
                    # 将all_nondominated_dis内个体，添加至overall_population_dis中，计算全局非支配解集
                    for indiv in all_nondominated_dis:
                        overall_population_dis.add(Individual(epoch=indiv.epoch, iteration=indiv.iteration,
                                                              model=indiv.model, code=indiv.code,
                                                              objectives=indiv.objectives))
                    # 将种群中的非支配解集保存至self.overall_nondominated_dis
                    self.overall_nondominated_dis = overall_population_dis.fronts

                # 计数下一迭代采样数量。采样分为：1、TopK采样 1/4；2、Random采样 3/4 - （非支配解集+变异解集）
                self.sample_number_dis = int((3/4)*self.sample_number - 2*len(population_dis.fronts))
                
                '''
                非支配解集队列交叉、变异产生后代
                '''
                front_queue_dis = copy.deepcopy(nondominated_queue_dis) # 复制一份帕累托前沿
                front_queue_dis_codes = front_queue_dis.get_all() # 获取帕累托前沿个体的code

                while not front_queue_dis.is_empty():
                    # 小于0.5时交叉,cross_or_mutate = True
                    # 大于0.5时变异,cross_or_mutate = False
                    cross_or_mutate = True if random.uniform(0, 1) < 0.5 else False
                                        
                    if cross_or_mutate == False: # 变异
                        '''cross_or_mutate == False时,变异''' 
                        chromosome_dis = front_queue_dis.pop()
                        while True:
                            mutated_chromosome_dis = self._mutate(chromosome_dis)
                            if mutated_chromosome_dis not in front_queue_dis_codes: # 确保变异个体不在帕累托前沿中
                                cross_and_mutate_queue_dis.insert(mutated_chromosome_dis)
                                front_queue_dis_codes.append(mutated_chromosome_dis)
                                break                        
                    else: 
                        '''cross_or_mutate == True时,
                           1、front_queue.size() >= 2,交叉,
                              但如果进入死循环(循环100次以上时),第一个基因变异，第二个基因放回
                           2、front_queue.size() == 1,依然变异
                        '''
                        if front_queue_dis.size() >= 2: # 交叉
                            chromosome1 = front_queue_dis.pop()
                            chromosome2 = front_queue_dis.pop()
                            count = 0 # 循环标记
                            while True:
                                count += 1
                                
                                crossovered_chromosome1, crossovered_chromosome2 = self._crossover(chromosome1, chromosome2)
                                if (crossovered_chromosome1 not in front_queue_dis_codes) and (crossovered_chromosome2 not in front_queue_dis_codes): # 确保交叉个体不在帕累托前沿中
                                    cross_and_mutate_queue_dis.insert(crossovered_chromosome1)
                                    cross_and_mutate_queue_dis.insert(crossovered_chromosome2)
                                    front_queue_dis_codes.append(crossovered_chromosome1)
                                    front_queue_dis_codes.append(crossovered_chromosome2)
                                    break

                                if count > self.while_circle_count: # 防止死循环
                                    while True:
                                        mutated_chromosome1 = self._mutate(chromosome1)
                                        if mutated_chromosome1 not in front_queue_dis_codes: # 确保变异个体不在帕累托前沿中
                                            cross_and_mutate_queue_dis.insert(mutated_chromosome1)
                                            front_queue_dis_codes.append(mutated_chromosome1)
                                            break # 跳出内层循环
                                    front_queue_dis.insert(chromosome2) # 将未交叉的个体重新放回队列
                                    break # 跳出外层循环
                        else: # 变异
                            chromosome = front_queue_dis.pop() 
                            while True:
                                mutated_chromosome = self._mutate(chromosome)
                                if mutated_chromosome not in front_queue_dis_codes: # 确保变异个体不在帕累托前沿中
                                    cross_and_mutate_queue_dis.insert(mutated_chromosome)
                                    front_queue_dis_codes.append(mutated_chromosome)
                                    break
                
                # 保存非支配解及其变异个体基因，用于下一个迭代中采样、评估
                survival_dis = []
                for _ in range(nondominated_queue_dis.size()):
                    survival_dis.append(nondominated_queue_dis.pop())
                    survival_dis.append(cross_and_mutate_queue_dis.pop())
                self.last_iteration_code_dis = survival_dis

                reward_credit_dis = population_dis.credit_reward()
                credit_probability_dis = self._credit_probability(reward_credits=reward_credit_dis,
                                                            codes=population_dis.codes)

                # Layer-Wise 更新 Theta
                for layer, (origin_theta, target_theta) in enumerate(zip(layer_wise_thetas_dis, credit_probability_dis)):
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(origin_theta, torch.from_numpy(target_theta).float().to(self.device_dis))

                    loss.backward(retain_graph=True)

                    layer_wise_theta_optimizers_dis[layer].step()
                    layer_wise_theta_schedulers_dis[layer].step()
                
                # 验证最优Discriminator子网络性能                
                if (iteration + 1) % self.print_freq == 0 and iteration != 0 or (iteration + 1) == len(loader):
                    vaild_start = time.time()
                    with torch.no_grad():
                        output_vaild = torch.empty(0).cuda(non_blocking=True)
                        labels_vaild = torch.empty(0).cuda(non_blocking=True)
                        for test_image, test_label in test_loader:                            
                            test_image, test_label = test_image.view(test_image.shape[0], -1, test_image.shape[1]).cuda(non_blocking=True), test_label.cuda(non_blocking=True)
                            out, _ = self.best_discriminator(test_image, self.temperature, 0, False)
                            output_vaild = torch.cat((output_vaild, out), dim=0)
                            label_one_hot = nn.functional.one_hot(test_label.type(torch.int64), num_classes=self.class_num).to(torch.float32).view(test_image.shape[0], 1, -1).cuda(non_blocking=True)
                            labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                        
                        vaild_accu = self._accuracy(output_vaild, labels_vaild)
                    vaild_end = time.time()
                    
                    # 每print_freq输出一次_intermediate_stats_logging   
                    self._intermediate_stats_logging(model_name='Discriminator', train_kind='theta',
                                                     accuracy=vaild_accu, step=iteration,
                                                     epoch=epoch, len_loader=len(loader),
                                                     time_consumption=vaild_end-vaild_start)
                    
            else: # 训练的是模型权重weights，则直接更新best_discriminator_supernet 
                
                # 验证Discriminator Supernet性能 
                if (iteration + 1) % self.print_freq == 0 and iteration != 0 or (iteration + 1) == len(loader):
                    vaild_start = time.time()
                    with torch.no_grad():
                        output_vaild = torch.empty(0).cuda(non_blocking=True)
                        labels_vaild = torch.empty(0).cuda(non_blocking=True)
                        for image, label in test_loader:                            
                            image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
                            out, _ = self.best_discriminator_supernet(image, self.temperature, 0, True)
                            output_vaild = torch.cat((output_vaild, out), dim=0)
                            label_one_hot = nn.functional.one_hot(label.type(torch.int64), num_classes=self.class_num).to(torch.float32).view(image.shape[0], 1, -1).cuda(non_blocking=True)
                            labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                        
                        vaild_accu = self._accuracy(output_vaild, labels_vaild)
                    vaild_end = time.time()
                    
                    # 每print_freq输出一次_intermediate_stats_logging
                    self._intermediate_stats_logging(model_name='Discriminator', train_kind='weight',
                                                     accuracy=vaild_accu, step=iteration,
                                                     epoch=epoch, len_loader=len(loader),
                                                     time_consumption=vaild_end-vaild_start)



            """""""""""""""
            Generator Supernet 训练 ———— 训练weights
            """""""""""""""
            # 按照反比例生成策略_weighted_random_int，构建生成样本的标签fake_label
            fake_label_gen = np.array([self._weighted_random_int() for _ in range(batch_size)])
            fake_label_gen = torch.from_numpy(fake_label_gen).to(int).cuda(non_blocking=True)
            fake_label_one_hot_gen = nn.functional.one_hot(fake_label_gen, num_classes=self.class_num).to(torch.float32).view(batch_size, 1, -1).cuda(non_blocking=True)
            
            label_gen = fake_label_gen.clone()
            image_gen = self._get_data_from_label(loader, self.class_number_count, label_gen)
            image_gen, label_gen = image_gen.view(batch_size, -1, image_gen.shape[1]).cuda(non_blocking=True), label_gen.cuda(non_blocking=True)
            
            # """
            # Generator优化器、学习率
            # """
            # thetas_params_gen = [param for name, param in generator_supernet.named_parameters() if 'thetas' in name]
            # params_except_thetas_gen = [param for param in generator_supernet.parameters() if not check_tensor_in_list(param, thetas_params_gen)]

            # weight_optimizer_gen = torch.optim.Adam(params=params_except_thetas_gen, 
            #                                    lr=CONFIG_SUPERNET['optimizer']['w_lr'], 
            #                                    weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
            # # Automatically optimizes learning rate
            # weight_scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer_gen, 
            #                                                              T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'], 
            #                                                              last_epoch=last_epoch)
                        
            generator_supernet.train()
            # 梯度归零
            weight_optimizer_gen.zero_grad()
            
            generate_fake_image, latency_to_accumulate_gen = generator_supernet(noise, fake_label_one_hot_dis, self.temperature, latency_to_accumulate_gen, True)
            generate_fake_image = generate_fake_image.view(batch_size, 1, -1).cuda(non_blocking=True)
            
            # self.best_discriminator_supernet.eval()
            outs_fake_gen, _ = discriminator_supernet(generate_fake_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), True)
            outs_real_gen, _ = discriminator_supernet(image_gen.detach(), self.temperature, latency_to_accumulate_dis, True)
            
            mean_fake_gen = outs_fake_gen.mean(dim=0, keepdim=True)
            mean_real_gen = outs_real_gen.mean(dim=0, keepdim=True)

            # R3GAN Generator的损失，只有softplus，没有R1+R2
            fake_loss_dis = nn.CrossEntropyLoss()(outs_fake_gen-mean_real_gen, label_gen)
            real_loss_dis = nn.CrossEntropyLoss()(outs_real_gen-mean_fake_gen, fake_label_gen)

            gen_loss = (fake_loss_dis + real_loss_dis).mean()
            
            gen_loss.backward(retain_graph=True)
            weight_optimizer_gen.step() # 更新模型参数
            weight_scheduler_gen.step() # 更新学习率
            
            self.best_generator_supernet = generator_supernet

            # 计算supernet的accuracy
            pred = torch.argmax(outs_fake_gen, dim=1)
            target = fake_label_gen
            
            gen_accuracy, _, _, _ = self._evaluation(model_kind='Generator', num_class=self.class_num, 
                                                          pred=pred, target=target)
            self.logger.info(f'Gen weights Iter:{iteration+1}, Accu:{gen_accuracy:.4f}, Loss:{gen_loss:.4f}')
            
        

            """
            Generator 子网采样、评估、选择  ———— 训练thetas
            """
            population_gen = Population()
            nondominated_queue_gen = IndividualQueue()
            cross_and_mutate_queue_gen = IndividualQueue()

            if weight_or_theta=='theta':
                # layer_wise_thetas_gen = [param for name, param in generator_supernet.named_parameters() if 'thetas' in name]
                # layer_wise_theta_optimizers_gen = [torch.optim.Adam(params=[layer], lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                #                                                     weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
                #                                                     for layer in layer_wise_thetas_gen]
                # # Automatically optimizes learning rate
                # layer_wise_theta_schedulers_gen = [torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1, 
                #                                                                           last_epoch=last_epoch)
                #                                                                           for optimizer in layer_wise_theta_optimizers_gen]

                for opt in layer_wise_theta_optimizers_gen:
                    opt.zero_grad()
                    
                """
                采样、评估、筛选子网络
                """
                # 开始权重训练的第1个Epoch只有Random采样
                if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                # if epoch == 0 and iteration == 0:
                    for i in range(self.sample_number):
                        # 采样Random个体,且规避重复
                        while True:
                            generator_random_sampled_model, _, generator_random_sampled_code = self._sample(generator_supernet, mode='Random')                     
                            if generator_random_sampled_code not in population_gen.codes:
                                break
                        # 计算Accuracy、Diversity
                        generator_random_sampled_model.eval()
                        with torch.no_grad():
                            random_sampled_gen_images = generator_random_sampled_model(noise, fake_label_one_hot_gen, self.temperature, 0, False)

                        random_sampled_gen_images = random_sampled_gen_images.view(batch_size, 1, -1).cuda(non_blocking=True)
                        
                        random_sampled_gen_images_output, random_sampled_pe_code_fake = self.best_discriminator(random_sampled_gen_images, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        # Accuracy
                        generator_random_sampled_model_accuracy = self._accuracy(output=random_sampled_gen_images_output, target=fake_label_gen)
                        # Diversity
                        random_sampled_inception_score = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(random_sampled_gen_images_output, dim=1), fake_label_one_hot_gen)
                        _, random_sampled_pe_code_validate = self.best_discriminator(validate_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        generator_random_sampled_model_diversity = self._diversity(inception_score=random_sampled_inception_score.cuda(non_blocking=True), 
                                                                                   pe_code_fake=random_sampled_pe_code_fake, 
                                                                                   pe_code_validate=random_sampled_pe_code_validate, 
                                                                                   validate_label=validate_label)
                        
                        population_gen.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=generator_random_sampled_model, code=generator_random_sampled_code,
                                                      objectives=[generator_random_sampled_model_diversity, generator_random_sampled_model_accuracy]))
                        
                else:
                    # 上一迭代非支配解集及其交叉、变异个体
                    for individual_gen_code in self.last_iteration_code_gen:

                        generator_survival_model, _ = self._sample_from_code(generator_supernet, individual_gen_code)

                        # 计算Accuracy、Diversity
                        generator_survival_model.eval()
                        with torch.no_grad():
                            survival_gen_images = generator_survival_model(noise, fake_label_one_hot_gen, self.temperature, 0, False)

                        survival_gen_images = survival_gen_images.view(batch_size, 1, -1).cuda(non_blocking=True)
                        
                        survival_gen_images_output, survival_pe_code_fake = self.best_discriminator(survival_gen_images, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        # Accuracy
                        generator_survival_model_accuracy = self._accuracy(output=survival_gen_images_output, target=fake_label_gen)
                        # Diversity
                        survival_inception_score = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(survival_gen_images_output, dim=1), fake_label_one_hot_gen)
                        _, survival_pe_code_validate = self.best_discriminator(validate_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        generator_survival_model_diversity = self._diversity(inception_score=survival_inception_score.cuda(non_blocking=True), 
                                                                             pe_code_fake=survival_pe_code_fake, 
                                                                             pe_code_validate=survival_pe_code_validate, 
                                                                             validate_label=validate_label)                     
                        
                        population_gen.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=generator_survival_model, code=individual_gen_code,
                                                      objectives=[generator_survival_model_diversity, generator_survival_model_accuracy]))
                        
                    # TopK采样
                    for _ in range(int(1/4*self.sample_number)):
                        # 采样TopK个体,且规避重复
                        while True:
                            generator_topk_sampled_model, _, generator_topk_sampled_code = self._sample(generator_supernet, mode='TopK')
                            if generator_topk_sampled_code not in population_gen.codes:
                                break

                        # 计算Accuracy、Diversity
                        generator_topk_sampled_model.eval()
                        with torch.no_grad():
                            topk_sampled_gen_images = generator_topk_sampled_model(noise, fake_label_one_hot_gen, self.temperature, 0, False)

                        topk_sampled_gen_images = topk_sampled_gen_images.view(batch_size, 1, -1).cuda(non_blocking=True)
                        
                        topk_sampled_gen_images_output, topk_sampled_pe_code_fake = self.best_discriminator(topk_sampled_gen_images, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        # Accuracy
                        generator_topk_sampled_model_accuracy = self._accuracy(output=topk_sampled_gen_images_output, target=fake_label_gen)
                        # Diversity
                        topk_sampled_inception_score = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(topk_sampled_gen_images_output, dim=1), fake_label_one_hot_gen)
                        _, topk_sampled_pe_code_validate = self.best_discriminator(validate_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        generator_topk_sampled_model_diversity = self._diversity(inception_score=topk_sampled_inception_score.cuda(non_blocking=True), 
                                                                                 pe_code_fake=topk_sampled_pe_code_fake, 
                                                                                 pe_code_validate=topk_sampled_pe_code_validate, 
                                                                                 validate_label=validate_label)                     
                        
                        population_gen.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=generator_topk_sampled_model, code=generator_topk_sampled_code,
                                                      objectives=[generator_topk_sampled_model_diversity, generator_topk_sampled_model_accuracy]))
                        
                    # Random采样
                    for _ in range(self.sample_number_gen):
                        # 采样Random个体,且规避重复
                        while True:
                            generator_random_sampled_model, _, generator_random_sampled_code = self._sample(generator_supernet, mode='Random')                     
                            if generator_random_sampled_code not in population_gen.codes:
                                break

                        # 计算Accuracy、Diversity
                        generator_random_sampled_model.eval()
                        with torch.no_grad():
                            random_sampled_gen_images = generator_random_sampled_model(noise, fake_label_one_hot_gen, self.temperature, 0, False)

                        random_sampled_gen_images = random_sampled_gen_images.view(batch_size, 1, -1).cuda(non_blocking=True)
                        
                        random_sampled_gen_images_output, random_sampled_pe_code_fake = self.best_discriminator(random_sampled_gen_images, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        # Accuracy
                        generator_random_sampled_model_accuracy = self._accuracy(output=random_sampled_gen_images_output, target=fake_label_gen)
                        # Diversity
                        random_sampled_inception_score = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(random_sampled_gen_images_output, dim=1), fake_label_one_hot_gen)
                        _, random_sampled_pe_code_validate = self.best_discriminator(validate_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                        generator_random_sampled_model_diversity = self._diversity(inception_score=random_sampled_inception_score.cuda(non_blocking=True), 
                                                                                   pe_code_fake=random_sampled_pe_code_fake, 
                                                                                   pe_code_validate=random_sampled_pe_code_validate, 
                                                                                   validate_label=validate_label)                     
                        
                        population_gen.add(Individual(epoch=epoch, iteration=iteration,
                                                      model=generator_random_sampled_model, code=generator_random_sampled_code,
                                                      objectives=[generator_random_sampled_model_diversity, generator_random_sampled_model_accuracy]))
                        
                # 保存非支配解集model,并构建非支配队列
                for individual in population_gen.fronts:
                    nondominated_queue_gen.insert(individual.code)
                    # self._model_save(model=individual,
                    #                  gen_or_dis='Generator', epoch=individual.epoch,
                    #                  iteration=individual.iteration, code=individual.code,
                    #                  objectives=individual.objectives, save_path=self.save_model_path)
                
                self.best_generator = population_gen.fronts[0].model
                # 计算全局最优非支配解，保存至self.overall_nondominated_gen
                if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                    self.overall_nondominated_gen = population_gen.fronts
                else:
                    overall_population_gen = Population()  # 构建种群，便于计算全局非支配解集
                    all_nondominated_gen = self.overall_nondominated_gen # 初始化全部非支配解集all_nondominated_gen，并将前一迭代的非支配解集添加进去
                    for indiv in population_gen.fronts: # 往all_nondominated_gen添加当前迭代的非支配解集
                        all_nondominated_gen.append(indiv)
                    # 将all_nondominated_gen内个体，添加至overall_population_gen中，计算全局非支配解集
                    for indiv in all_nondominated_gen:
                        overall_population_gen.add(Individual(epoch=indiv.epoch, iteration=indiv.iteration,
                                                              model=indiv.model, code=indiv.code,
                                                              objectives=indiv.objectives))
                    # 将种群中的非支配解集保存至self.overall_nondominated_gen
                    self.overall_nondominated_gen = overall_population_gen.fronts

                # 计数下一迭代采样数量。采样分为：1、TopK采样 1/4；2、Random采样 3/4 - （非支配解集+变异解集）
                self.sample_number_gen = int((3/4)*self.sample_number - 2*len(population_gen.fronts))
                
                '''
                非支配解集队列交叉、变异产生后代
                '''
                front_queue_gen = copy.deepcopy(nondominated_queue_gen)  # 复制一份帕累托前沿
                front_queue_gen_codes = front_queue_gen.get_all() # 获取帕累托前沿个体的code
                
                while not front_queue_gen.is_empty():
                    # 小于0.5时交叉,cross_or_mutate = True
                    # 大于0.5时变异,cross_or_mutate = False
                    cross_or_mutate = True if random.uniform(0, 1) < 0.5 else False
                    if cross_or_mutate == False: # 变异
                        '''cross_or_mutate == False时,变异''' 
                        chromosome = front_queue_gen.pop()
                        while True:
                            mutated_chromosome = self._mutate(chromosome)
                            if mutated_chromosome not in front_queue_gen_codes: # 确保变异个体不在帕累托前沿中
                                cross_and_mutate_queue_gen.insert(mutated_chromosome)
                                front_queue_gen_codes.append(mutated_chromosome)
                                break
                    else: 
                        '''cross_or_mutate == True时,
                           1、front_queue.size >= 2,交叉,
                              但如果进入死循环(循环100次以上时),第一个基因变异，第二个基因放回
                           2、front_queue.size == 1,依然变异
                        '''
                        if front_queue_gen.size() >= 2: # 交叉
                            chromosome1 = front_queue_gen.pop()
                            chromosome2 = front_queue_gen.pop()
                            
                            count = 0 # 循环标记
                            while True: # 外层循环
                                count += 1

                                crossovered_chromosome1, crossovered_chromosome2 = self._crossover(chromosome1, chromosome2)
                                if (crossovered_chromosome1 not in front_queue_gen_codes) and (crossovered_chromosome2 not in front_queue_gen_codes): # 确保交叉个体不在帕累托前沿中
                                    cross_and_mutate_queue_gen.insert(crossovered_chromosome1)
                                    cross_and_mutate_queue_gen.insert(crossovered_chromosome2)
                                    front_queue_gen_codes.append(crossovered_chromosome1)
                                    front_queue_gen_codes.append(crossovered_chromosome2)
                                    break

                                if count > self.while_circle_count: # 防止死循环
                                    while True: # 内层循环
                                        mutated_chromosome1 = self._mutate(chromosome1)
                                        if mutated_chromosome1 not in front_queue_gen_codes: # 确保变异个体不在帕累托前沿中
                                            cross_and_mutate_queue_gen.insert(mutated_chromosome1)
                                            front_queue_gen_codes.append(mutated_chromosome1)
                                            break # 跳出内层循环
                                    front_queue_gen.insert(chromosome2) # 将未交叉的个体重新放回队列
                                    break # 跳出外层循环
                        else: # 变异
                            chromosome = front_queue_gen.pop()
                            
                            while True:
                                mutated_chromosome = self._mutate(chromosome)
                                if mutated_chromosome not in front_queue_gen_codes: # 确保变异个体不在帕累托前沿中
                                    cross_and_mutate_queue_gen.insert(mutated_chromosome)
                                    front_queue_gen_codes.append(mutated_chromosome)
                                    break
                            
                # 保存非支配解及其变异个体基因，用于下一个迭代中采样、评估
                survival_gen = []
                for _ in range(nondominated_queue_gen.size()):
                    survival_gen.append(nondominated_queue_gen.pop())
                    survival_gen.append(cross_and_mutate_queue_gen.pop())
                self.last_iteration_code_gen = survival_gen

                reward_credit_gen = population_gen.credit_reward()
                credit_probability_gen = self._credit_probability(reward_credits=reward_credit_gen,
                                                            codes=population_gen.codes)
                
                # Layer-Wise 更新 Theta
                for layer, (origin_theta, target_theta) in enumerate(zip(layer_wise_thetas_gen, credit_probability_gen)):
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(origin_theta, torch.from_numpy(target_theta).float().to(self.device_gen))

                    loss.backward(retain_graph=True)

                    layer_wise_theta_optimizers_gen[layer].step()
                    layer_wise_theta_schedulers_gen[layer].step()
                
                # 每print_freq输出一次_intermediate_stats_logging
                self._intermediate_stats_logging(model_name='Generator',
                                                 train_kind='theta',
                                                 accuracy=population_gen.fronts[0].objectives[1],
                                                 step=iteration,
                                                 epoch=epoch,
                                                 len_loader=len(loader),
                                                 time_consumption=0)
            
            # else:
            #     self.best_generator_supernet = generator_supernet
                
                # if (iteration + 1) % self.print_freq == 0 and iteration != 0 or (iteration + 1) == len(loader):
                #     vaild_start = time.time()
                #     with torch.no_grad():
                #         output_vaild = torch.empty(0).cuda(non_blocking=True)
                #         labels_vaild = torch.empty(0).cuda(non_blocking=True)
                #         for image, label in test_loader:                            
                #             image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
                #             out, _ = self.best_discriminator_supernet(image, self.temperature, 0, True)
                #             output_vaild = torch.cat((output_vaild, out), dim=0)
                #             label_one_hot = nn.functional.one_hot(label.type(torch.int64), num_classes=self.class_num).to(torch.float32).view(image.shape[0], 1, -1).cuda(non_blocking=True)
                #             labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                        
                #         vaild_accu = self._accuracy(output_vaild, labels_vaild)
                #     vaild_end = time.time()

                #     # 每print_freq输出一次_intermediate_stats_logging
                #     self._intermediate_stats_logging(model_name='Generator',
                #                                     train_kind='weight',
                #                                     accuracy=gen_accuracy_list,
                #                                     step=iteration,
                #                                     epoch=epoch,
                #                                     len_loader=len(loader),
                #                                     time_consumption=0)

            iteration += 1
            self.logger.info('')
            
            # if weight_or_theta == 'weight':
            #     if iteration == CONFIG_SUPERNET['train_settings']['print_freq']*2:
            #         break
            # else:
            #     if iteration == CONFIG_SUPERNET['train_settings']['print_freq']*2:
            #         break
    
    def _cosine_decay_with_warmup(self, cur_nimg, base_value, total_nimg, final_value=0.0, warmup_value=0.0, warmup_nimg=0, hold_base_value_nimg=0):
        decay = 0.5 * (1 + np.cos(np.pi * (cur_nimg - warmup_nimg - hold_base_value_nimg) / float(total_nimg - warmup_nimg - hold_base_value_nimg)))
        cur_value = base_value + (1 - decay) * (final_value - base_value)
        if hold_base_value_nimg > 0:
            cur_value = np.where(cur_nimg > warmup_nimg + hold_base_value_nimg, cur_value, base_value)
        if warmup_nimg > 0:
            slope = (base_value - warmup_value) / warmup_nimg
            warmup_v = slope * cur_nimg + warmup_value
            cur_value = np.where(cur_nimg < warmup_nimg, warmup_v, cur_value)
        return float(np.where(cur_nimg > total_nimg, final_value, cur_value))

    def _zero_centered_gradient_penalty(self, Samples, Critics):
        #  计算梯度的L2范数的平方
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        #  - 将梯度张量展平 (除了batch维度)
        Gradient = Gradient.view(Gradient.size(0), -1)
        #  - 计算每个样本梯度的L2范数的平方: sum(grad^2)
        #  - 在整个batch上取平均
        penalty = torch.mean(torch.sum(Gradient ** 2, dim=1))

        return penalty

    def _weighted_random_int(self):
        weights = [1/math.sqrt(count + 1) for count in self.class_number_count]
        total = sum(weights)
        r = random.uniform(0, total)
        s = 0
        for i, w in enumerate(self.invert_ratio):
            s += w
            if r < s:
                return i
    
    def _accuracy(self, output, target):
        # device = {'Discriminator':self.device_dis, 'Generator':self.device_gen}
        if len(target.shape) >= 2:
            target = target.reshape(target.shape[0], -1)
            target = torch.argmax(target, axis=1)

        output = torch.argmax(output, axis=1)
        
        accuracy = accuracy_score(output.cpu(), target.cpu())
            
        return accuracy
    
    def _calculate_confusion_matrix(self, num_class, pred, target):
        
        confusion_matrix = torch.zeros(size=(num_class, num_class)).cuda()
        for i in range(len(pred)):
            confusion_matrix[pred[i], target[i]] += 1        
        
        return confusion_matrix

    def _evaluation(self, model_kind, num_class, pred, target):
        
        device = {'Discriminator':self.device_dis, 'Generator':self.device_gen}
        pred = pred.to(device[model_kind])
        target = target.to(device[model_kind])
        
        confusion_matrix = self._calculate_confusion_matrix(num_class, pred, target)
        accuracy = sum(confusion_matrix.diag()) / len(pred)
        precision = confusion_matrix.diag() / (confusion_matrix.sum(1) + 1e-8)
        recall = confusion_matrix.diag() / (confusion_matrix.sum(0) + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        return accuracy, precision, recall, f1_score
    
    def _diversity(self, inception_score, pe_code_fake, pe_code_validate, validate_label):
        # 把数据放入GPU计算
        inception_score = inception_score.to(self.device_gen)
        pe_code_fake = pe_code_fake.to(self.device_gen)
        pe_code_validate = pe_code_validate.to(self.device_gen)
        validate_label = validate_label.to(self.device_gen)
        
        # 生成数据PositionalEncoding取平均
        average_pe_fake = torch.mean(pe_code_fake, axis=0)
        # 验证集中正常类型数据(标签为0的数据，即not label)的平均PositionalEncoding初始化
        average_pe_validate = torch.zeros(pe_code_validate[0].shape, dtype=torch.float32).cuda(non_blocking=True).to(self.device_gen)
        for code, label in zip(pe_code_validate, validate_label):
            if not label:
                average_pe_validate += code
        average_pe_validate = average_pe_validate / sum(validate_label)
        # diversity=e^inception_score / mean((average_pe_validate - average_pe_fake)**2)
        diversity = (inception_score / torch.mean(pow(abs(average_pe_validate - average_pe_fake), 2))).item()

        return diversity
      
    def _intermediate_stats_logging(self, model_name, train_kind, accuracy, step, epoch, len_loader, time_consumption):        
        if (step > 0) and ((step + 1) % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(
                model_name +' '+ train_kind +
                ": Epo:[{}/{}], Iter:[{}/{}], Accuracy:{:.4f}, Time:{:.4f}".format(
                epoch + 1, self.cnt_epochs, step + 1, len_loader, accuracy, time_consumption))
    
    def _sample(self, model, mode, k=2):
        # print(f'{model.module.__class__.__name__}/{mode}')
        sampled_model = copy.deepcopy(model)
        # print(f'---------------------Original Model:\n{sampled_model}\n')
        ops_names = [op_name for op_name in self.lookup_table.lookup_table_operations]
        # cnt_ops = len(ops_names)

        sampled_latency = 0
        arch_operations=[]
        chosen_code = []
        
        for index, layer in enumerate(sampled_model.module.stages_to_search): 
            if mode == 'Max':
                optimal_ops_index = np.argmax(layer.thetas.detach().cpu().numpy())
            elif mode == 'TopK':
                _, indices = torch.topk(layer.thetas, k)
                optimal_ops_index = np.random.choice(indices.detach().cpu().numpy(), 1)[0]
            elif mode == 'Random':
                optimal_ops_index = np.random.randint(0, len(self.lookup_table.lookup_table_operations))
            
            chosen_code.append(int(optimal_ops_index))
            # Latency Calculation
            sampled_latency += self.lookup_table.lookup_table_latency[index][ops_names[optimal_ops_index]]
            # Operation Chosen
            arch_operations.append(layer.ops[optimal_ops_index])
            
        sampled_model.module.stages_to_search = nn.Sequential(*arch_operations)
        
        return sampled_model, sampled_latency, chosen_code
            
    def _sample_from_code(self, model, code):

        sampled_model = copy.deepcopy(model)
        # print(f'---------------------Original Model:\n{sampled_model}\n')
        ops_names = [op_name for op_name in self.lookup_table.lookup_table_operations]

        sampled_latency = 0
        arch_operations=[]
        
        for i, (index, layer) in enumerate(zip(code, sampled_model.module.stages_to_search)): 
            # Latency Calculation
            sampled_latency += self.lookup_table.lookup_table_latency[i][ops_names[index]]
            # Operation Chosen
            arch_operations.append(layer.ops[index])
            
        sampled_model.module.stages_to_search = nn.Sequential(*arch_operations)
        
        return sampled_model, sampled_latency

    def _credit_probability(self, reward_credits, codes):
        total_credits = np.zeros((6, len(self.lookup_table.lookup_table_operations))) # 初始化，归零
        
        for idx in reward_credits: # 遍历rewared_credits健值对
            for layer_idx, opt_idx in enumerate(codes[idx]): # 遍历子网基因点
                total_credits[layer_idx, opt_idx] += reward_credits[idx] # 对应位置加上credit
        
        return total_credits/total_credits.sum(axis=1)[0] # 求和后处以每一列/层的credit总和，返回占比

    def _theta_array_value(self, model, model_kind):
        if model_kind == 'Discriminator':
            thetas_params_dis = [param for name, param in model.named_parameters() if 'thetas' in name]
            # thetas_array_dis = np.array([line.detach().cpu().numpy() for line in thetas_params_dis])
            return thetas_params_dis
        elif model_kind == 'Generator':
            thetas_params_gen = [param for name, param in model.named_parameters() if 'thetas' in name]
            # thetas_array_gen = np.array([line.detach().cpu().numpy() for line in thetas_params_gen])
            return thetas_params_gen

    def _theta_reward_update(self, model, code_count, lr, device):
        
        for layer, count in zip(model.module.stages_to_search, code_count):
            # print(f'-- Before:{sum(layer.thetas.detach().cpu().numpy())}\n{layer.thetas.detach().cpu().numpy()}')
            new_theta = np.array(layer.thetas.detach().cpu().numpy() + np.array(count * lr))
            # new_theta = self._norm(new_theta)        
            layer.thetas.data = torch.nn.Parameter(torch.from_numpy(new_theta).type(torch.float)).to(device)
            # print(f'--  After:{sum(layer.thetas.detach().cpu().numpy())}\n{layer.thetas.detach().cpu().numpy()}')
            
    def _norm(self, value):
        # print(f'-- Before:{sum(value)}\n{value}')
        row_sum = sum(value)
        for indice, item in enumerate(value):
            value[indice] = item/row_sum
        # print(f'--  After:\n{value}')
        return value
    
    def _crossover(self, chromosome1, chromosome2):
        code1 = copy.deepcopy(chromosome1)
        code2 = copy.deepcopy(chromosome2)
        
        exchange_length = random.randint(2,len(code1)-1)
        # 随机取start1和start2
        start1, start2 = random.randint(0,len(code1)-exchange_length), random.randint(0,len(code1)-exchange_length)
        # 对应位置交换            
        for i in range(exchange_length):
            code1[start1+i], code2[start2+i] = code2[start2+i], code1[start1+i]
        return code1, code2
    
    def _mutate(self, chromosome):
        code = copy.deepcopy(chromosome)
        
        mutate_index = random.randint(0,len(code)-1)
        # 当前第mutate_index上的值为current_num
        current_num = code[mutate_index]
        while current_num == code[mutate_index]:
            code[mutate_index] = random.randint(0, len(self.lookup_table.lookup_table_operations)-1)
        
        return code
               
    def _model_save(self, model, gen_or_dis, epoch, iteration, code, objectives, save_path):
        save_path = save_path+'/'+gen_or_dis+'/'+f'Epoch{epoch+1}'+'/'+f'Iter{iteration+1}'
        os.makedirs(save_path, exist_ok=True) # Create directories if needed, ignore if already exist
        
        objective = ''
        if gen_or_dis == 'Discriminator':
            objectives_name = ['Accuracy', 'Latency']
        elif gen_or_dis == 'Generator':
            objectives_name = ['Accuracy', 'Latency', 'Diversity']
        for obj, name in zip(objectives, objectives_name):
            objective += (name + ':' + str(obj) + ' ')
            
        model_name = str(code) + '-' + objective + '.pt'
        model_path = os.path.join(save_path, model_name)
        
        # print(f'-- Saving {gen_or_dis} - {model_path}')
            
        torch.save(model, model_path)
        
    def _model_load(self, gen_or_dis, epoch, iteration, code, objectives, save_path):
        save_path = save_path+'/'+gen_or_dis+'/'+f'Epoch{epoch+1}'+'/'+f'Iter{iteration+1}'
        
        objective = ''
        if gen_or_dis == 'Discriminator':
            objectives_name = ['Accuracy', 'Latency']
        elif gen_or_dis == 'Generator':
            objectives_name = ['Accuracy', 'Latency', 'Diversity']
        for obj, name in zip(objectives, objectives_name):
            objective += (name + ':' + str(obj) + ' ')
            
        model_name = str(code) + '-' + objective + '.pt'
        model_path = os.path.join(save_path, model_name)

        model = torch.load(model_path)
        
        return model
    
    def _get_data_from_label(self, loader, class_number_count, fake_label):
       
        indices = []
        for item in fake_label:
            # 在每个类别的范围内随机采样一个索引
            indices.append(self.class_index[item][random.randint(0, len(self.class_index[item])-1)])
            
        # 按照采样序号构建一个batch的真实数据
        sampled_real_data = loader.dataset.data[indices]

        return sampled_real_data