import numpy as np

class Individual(object):
    """Individual类"""
    def __init__(self, epoch, iteration, model, code, objectives=[], dominated_count=0, dominated_solutions=[], rank=0, crowding_distance=0):
        self.epoch = epoch
        self.iteration = iteration
        self.model= model
        self.code = code
        self.objectives = objectives        
        self.dominated_count = dominated_count
        self.dominated_solutions = dominated_solutions
        self.rank = rank
        self.crowding_distance = crowding_distance
        
    def dominates(self, objectives): # 当前个体支配XXX个体
        dominates = True
        # if self.objectives != objectives:
        if not np.array_equal(self.objectives, objectives):
            for item1, item2 in zip(self.objectives, objectives):
                if item1 < item2:
                    dominates = False
        else:
            dominates = True
        # print(f'{self.objectives}vs{objective} -- Dominated:{dominated}')
        return dominates
    
    def dominated_by(self, objectives): # 当前个体被XXX个体支配
        dominated = True
        # if self.objectives != objectives:
        if not np.array_equal(self.objectives, objectives):
            for item1, item2 in zip(objectives, self.objectives):
                if item1 < item2:
                    dominated = False
        else:
            dominated = False
        # print(f'{self.objectives}vs{objective} -- Dominated:{dominated}')
        return dominated

    def get_model_info(self): # 个体信息
        return self.epoch, self.iteration, self.code, self.objectives
    
       
class Population(object):
    """Represents population - a group of Individuals,
    can merge with another population"""
    def __init__(self):
        
        self.fronts = []       # 非支配解
        self.sorting = {}      # 按层排列的NSGA2排序结果

        self.objectives = []   # 多目标评估结果
        self.codes = []        # 个体编码
        self.dominates_number = [] # 支配解的数量
        self.dominated_number = [] # 被支配解的数量
       
    def add(self, individual): # 增加Population新个体
        self.objectives.append(individual.objectives) # 增加新个体多目标评估结果
        self.codes.append(individual.code) # 增加新个体code

        if len(self.objectives) == 1: # 如果是第一个个体，直接添加新个体到非支配解fronts中，排序为{0:0，}
            self.fronts.append(individual) # 将第一个个体添加到非支配解fronts中
            self.sorting = {0:[0]} # 唯一个体在第0层，编号为0
            self.dominates_number.append(0) # 支配解的数量
            self.dominated_number.append(0) # 被支配解的数量

        if len(self.objectives) >= 2: # 如果是第一个个体，直接添加到非支配解fronts中
            new_sorting = self.fast_nondominated_sort(individual) # 对所有个体进行NSGA2排序

            new_front = [] # 新非支配前沿
            for new_front_indvidual_idx in new_sorting[0]: # 遍历新的非支配解集
                # 将支配解数量为0的个体添加到非支配解fronts中
                # if   : self.sorting[0]为空，即self.sorting为空，将新增个体插入前沿
                # elif : 不在旧的非支配解集中，只能是新增的个体，插入新前沿
                # else : 在旧的非支配解集中，寻找其下标，插入新前沿
                if not self.sorting:
                    new_front.append(individual)
                elif new_front_indvidual_idx not in self.sorting[0]: 
                    new_front.append(individual)
                else: 
                    fronts_idx = self.sorting[0].index(new_front_indvidual_idx)
                    new_front.append(self.fronts[fronts_idx])
            # 更新新前沿和NSGA2排序结果
            self.fronts = new_front
            self.sorting = new_sorting
        
    def fast_nondominated_sort(self, individual): # 快速非支配排序
        ranks = {}
        sorted_idx = {}

        self.dominates_number.append(0)        
        self.dominated_number.append(0)

        # 遍历当前种群中的每个个体，计算支配数和被支配数
        for index, other_individual in enumerate(self.objectives[0:-1]):
            # 如果新增个体individual支配other_individual，
            # individual支配解数量+1，则other_individual被支配解数量+1
            if individual.dominates(other_individual):
                self.dominates_number[-1] += 1 # individual支配解数量+1
                self.dominated_number[index] += 1 # other_individual被支配解数量+1
            # 如果other_individual支配新增个体individual，
            # 则individual被支配解数量+1,other_individual支配解数量+1
            elif individual.dominated_by(other_individual):
                self.dominated_number[-1] += 1 # individual被支配解数量+1
                self.dominates_number[index] += 1 # other_individual支配解数量+1
        
        # 初始化max(self.dominated_number)层的帕累托层ranks
        # 初始化排名索引sorted_idx
        for i in range(max(self.dominated_number)+1):
            ranks[i] = []
            

        # 遍历当前种群中的每个个体，计算帕累托层编号ranks
        for idx, dominated in enumerate(self.dominated_number):
            ranks[dominated].append(idx) # 将支配解数量为dominated的个体索引添加到ranks中
          
        for layer_idx in ranks:
            if len(ranks[layer_idx]) >= 2:
                layer_objectives = [self.objectives[i] for i in ranks[layer_idx]] # 获取当前帕累托层的所有个体的多目标评估值
                crowding_distance = self.crowding_distance(layer_objectives) # 计算每个帕累托层的拥挤距离
                
                distence_idx = np.argsort(-crowding_distance) # 获取拥挤距离排序后的索引(降序(-np.array为倒序))
                sorted_idx[layer_idx] = [ranks[layer_idx][i] for i in distence_idx] # 将拥挤距离排序后的索引添加到sorted_idx中
            else:
                sorted_idx[layer_idx] = ranks[layer_idx]

        return sorted_idx

    def crowding_distance(self, front): # 计算一个front中每个个体的拥挤距离
        """V1.标准的拥挤度计算,改进端点不足"""
        # front_objectives = np.array(front)  # 将front转换为numpy数组
        # n, num_objectives = front_objectives.shape
        # crowding_distance = np.zeros(n)  # 拥挤距离初始化为0

        # # 对每个目标分别计算
        # for m in range(num_objectives):
        #     # 对目标m进行排序，返回排序后索引
        #     sorted_idx = np.argsort(front_objectives[:, m])
        #     # 排序后的目标值
        #     sorted_obj = front_objectives[sorted_idx, m]
            
        #     # 归一化分母，避免分母为0
        #     obj_range = sorted_obj[-1] - sorted_obj[0]
        #     if obj_range == 0:
        #         obj_range = 1e-12  # 防止除零

        #     # 边界点只与唯一邻居比较，不赋无穷
        #     # 最左点
        #     crowding_distance[sorted_idx[0]] += (sorted_obj[1] - sorted_obj[0]) / obj_range
        #     # 最右点
        #     crowding_distance[sorted_idx[-1]] += (sorted_obj[-1] - sorted_obj[-2]) / obj_range
        #     # 中间所有点
        #     for i in range(1, n-1):
        #         crowding_distance[sorted_idx[i]] += (sorted_obj[i+1] - sorted_obj[i-1]) / obj_range
        #     return crowding_distance
        
        """V1.1、标准的拥挤度计算,改进端点不足+贡献程度不足"""
        front_objectives = np.array(front)  # 将front转换为numpy数组
        n, num_objectives = front_objectives.shape # 　
        crowding_distance = np.zeros(n)  # 拥挤距离初始化为0

        # 对每个目标分别计算
        for m in range(num_objectives):
            # 对目标m进行排序，返回排序后递增索引
            sorted_idx = np.argsort(front_objectives[:, m])
            # 排序后的目标值
            sorted_obj = front_objectives[sorted_idx, m]
            
            obj_min = sorted_obj[0]  # 最小值
            obj_max = sorted_obj[-1]  # 最大值

            # 归一化分母，避免分母为0
            obj_range = obj_max - obj_min
            # obj_range = np.std(front_objectives[:, m])
            if obj_range == 0:
                obj_range = 1e-12  # 防止除零

            # 边界点只与唯一邻居比较，不赋无穷
            # 最左点
            crowding_distance[sorted_idx[0]] += ((sorted_obj[1] - sorted_obj[0]) / obj_range ) * ((sorted_obj[0] - obj_min) / obj_range)
            # 最右点
            crowding_distance[sorted_idx[-1]] += ((sorted_obj[-1] - sorted_obj[-2]) / obj_range) * ((sorted_obj[-1] - obj_min) / obj_range)
            # 中间所有点
            for i in range(1, n-1):
                crowding_distance[sorted_idx[i]] += (((sorted_obj[i+1] - sorted_obj[i-1]) / 2) / obj_range) * ((sorted_obj[i] - obj_min) / obj_range)
                
        return crowding_distance

        # """V2.基于超体积贡献的拥挤度计算"""
        # front_objectives = np.array(front)
        # n = len(front_objectives)
        # crowding_distance = np.zeros(n)
        
        # if n <= 1:
        #     return np.full(n, float('inf'))
        
        # # 计算每个点被移除后超体积的减少量
        # for i in range(n):
        #     # 创建不包含第i个点的front
        #     remaining_front = np.delete(front_objectives, i, axis=0)
            
        #     # 简化的超体积贡献计算（基于支配空间）
        #     contribution = 1.0
        #     for j, other_point in enumerate(remaining_front):
        #         # 计算点i相对于其他点的支配空间
        #         if np.all(front_objectives[i] >= other_point):
        #             diff = front_objectives[i] - other_point
        #             contribution *= np.prod(diff)
            
        #     crowding_distance[i] = contribution
        
        # return crowding_distance

    def credit_reward(self): # 给NSGA2排序完毕的sorting积分:第一名100;第二名99;……依此类推
        reward = {}
        credit = []
        for idx in self.sorting:
            if self.sorting[idx]:
                for i in self.sorting[idx]:
                    credit.append(i)
                    
        for idx, item in enumerate(credit[::-1]): # 遍历翻转后的credit列表
            reward[item] = idx + 1  # 从1开始编号
        return reward
    
    def __len__(self): # 返回Population的长度
        
        return len(self.objectives)
        
    def __iter__(self): # 让 Population 类的实例可以用于for循环遍历，（如 for ind in pop:），直接遍历所有个体
        return self.individuals.__iter__()
    

class IndividualQueue:
    """普通先进先出队列，用于管理Individual个体"""
    def __init__(self):
        self.queue = []

    def insert(self, individual): # 入队操作：将一个Individual个体加入队尾
        self.queue.append(individual)

    def pop(self): # 出队操作：移除并返回队首的Individual个体
        if self.is_empty():
            raise IndexError("队列为空，无法出队")
        return self.queue.pop(0)

    def top(self): # 查看队首元素，但不移除它
        if self.is_empty():
            return None
        return self.queue[0]

    def is_empty(self): # 判断队列是否为空
        return len(self.queue) == 0

    def size(self): # 返回队列的长度
        return len(self.queue)

    def clear(self): # 清空队列
        self.queue = []

    def get_all(self): # 返回所有个体的列表
        return self.queue[:]

    def __iter__(self): # 支持迭代
        return iter(self.queue)