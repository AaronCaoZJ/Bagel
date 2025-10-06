"""
Utility for Speca
"""

from typing import Dict 
import torch
import math


#########################################
# Error calculation functions
#########################################
class ErrorCalculator:
    """
    误差计算工具类, 支持L1、L2、相对误差和cosine相似度误差
    usage:
        error_calculator = ErrorCalculator(eps=1e-x)
        error = error_calculator.all(x, full_x)
    """

    def __init__(self, eps=1e-10):
        self.eps = eps

    def l1(self, x, full_x):
        """计算L1误差 (平均绝对误差)"""
        return torch.abs(x - full_x).mean().item()

    def l2(self, x, full_x):
        """计算L2误差 (均方根误差)"""
        return torch.sqrt(torch.mean((x - full_x) ** 2)).item()

    def relative_l1(self, x, full_x):
        """计算相对L1误差"""
        error = torch.abs(x - full_x) / (torch.abs(full_x) + self.eps)
        return error.mean().item()

    def relative_l2(self, x, full_x):
        """计算相对L2误差"""
        error = torch.abs(x - full_x) / (torch.abs(full_x) + self.eps)
        return torch.sqrt(torch.mean(error ** 2)).item()

    def cosine_similarity(self, x, full_x):
        """计算cosine相似度误差 (1 - cosine_similarity)"""
        x_flat = x.view(x.size(0), -1)
        full_x_flat = full_x.view(full_x.size(0), -1)
        cosine_sim = torch.nn.functional.cosine_similarity(x_flat, full_x_flat, dim=1)
        return (1 - cosine_sim.mean()).item()

    def all(self, x, full_x):
        """计算所有误差指标，返回字典"""
        return {
            'l1': self.l1(x, full_x),
            'l2': self.l2(x, full_x),
            'relative_l1': self.relative_l1(x, full_x),
            'relative_l2': self.relative_l2(x, full_x),
            'cosine_similarity': self.cosine_similarity(x, full_x)
        }


# #########################################
# # TaylorSeer functions
# #########################################
# def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
#     """
#     Compute derivative approximation.
    
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

#     updated_taylor_factors = {}
#     updated_taylor_factors[0] = feature

#     for i in range(cache_dic['max_order']):
#         if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
#             updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
#         else:
#             break

# def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
#     """
#     Compute Taylor expansion error.
    
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     x = current['step'] - current['activated_steps'][-1]
#     #x = current['t'] - current['activated_times'][-1]
#     output = 0

#     for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
#         output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)

#     return output

# # cache for taylor factors
# def taylor_cache_init(cache_dic: Dict, current: Dict):
#     """
#     Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
#     :param cache_dic: Cache dictionary
#     :param current: Information of the current step
#     """
#     if (current['step'] == 0) and (cache_dic['taylor_cache']):
#         cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


#########################################
# Type control functions
#########################################
def speca_cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    min_taylor_steps = cache_dic['min_taylor_steps']
    max_taylor_steps = cache_dic['max_taylor_steps']

    if current['last_type'] == 'full':
        current['type'] = 'Taylor'
        cache_dic['taylor_step_counter'] = 1  
        cache_dic['check'] = False
        current['last_layer_error'] = None
    else:
        # if (cache_dic['fresh_ratio'] == 0.0) and (not cache_dic['taylor_cache']):
        #     # 仅第0步full，后续均使用预测
        #     first_steps = (current['step'] == 0)
        # 初始几步full，后续taylorseer
        first_steps = (current['step'] < cache_dic['first_enhance'])
        reached_max_taylor = (cache_dic['taylor_step_counter'] >= max_taylor_steps)
        progress = (current['num_steps'] - current['step']) / current['num_steps']
        base_threshold = cache_dic['base_threshold']
        decay_rate = cache_dic['decay_rate']
        threshold = base_threshold * (decay_rate ** progress)
        threshold = max(threshold, 0.01)

        if cache_dic['taylor_step_counter'] >= min_taylor_steps:
            cache_dic['check'] = True
        else:
            cache_dic['check'] = False

        error_too_large = current.get('last_layer_error') is not None and current.get('last_layer_error') > threshold

        if first_steps or reached_max_taylor or (error_too_large and cache_dic['check']):
            current['type'] = 'full'
            cache_dic['taylor_step_counter'] = 0
            cache_dic['full_count'] += 1
        
        else:
            cache_dic['taylor_step_counter'] < min_taylor_steps
            current['type'] = 'Taylor'
            cache_dic['taylor_step_counter'] += 1
        
    current['last_type'] = current['type']

    if current['type'] == 'full':
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
    else:
        cache_dic['cache_counter'] += 1


##########################################
# System level cache init function
##########################################
def cache_init(
        self, 
        num_steps: int, 
        # taylor_fresh_threshold=4,
        taylor_first_enhance=5,
        taylor_max_order=6,
        speca_base_threshold=0.1,
        speca_decay_rate=0.9,
        speca_min_taylor_steps=2,
        speca_max_taylor_steps=5,
        speca_error_metric='l1'
    ):
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache[-1]['layers_stream']={}

    for j in range(len(self.language_model.model.layers)):
        cache[-1]['layers_stream'][j] = {}
        cache_index[-1][j] = {}

    cache_dic['Delta-DiT'] = False
    cache_dic['cache_type'] = 'random'
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = 'ToCa' 
    cache_dic['fresh_ratio'] = 0.0
    cache_dic['soft_fresh_weight'] = 0.0

    # taylorseer parameters
    cache_dic['taylor_cache'] = True
    cache_dic['max_order'] = taylor_max_order
    # cache_dic['fresh_threshold'] = taylor_fresh_threshold
    cache_dic['first_enhance'] = taylor_first_enhance

    cache_dic['cache_counter'] = 0
    cache_dic['taylor_step_counter']  = 0
    cache_dic['full_count'] = 0

    # speca parameters
    cache_dic['base_threshold']  = speca_base_threshold
    cache_dic['decay_rate']  = speca_decay_rate
    cache_dic['min_taylor_steps']  = speca_min_taylor_steps
    cache_dic['max_taylor_steps']  = speca_max_taylor_steps
    cache_dic['error_metric'] = speca_error_metric
    cache_dic['check'] = False

    # current step information
    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = num_steps
    current['last_type'] = 'None'
    current['last_layer_error'] = 0.0

    return cache_dic, current
