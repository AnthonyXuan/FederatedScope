# 需要在criterion层面，对不同样本进行区分
# 需要增加一（两）个context变量，存放后门样本和干净样本
# 根据通信轮数进行方案两个阶段的区别


import copy
import logging

import torch

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from typing import Type

logger = logging.getLogger(__name__)

def wrap_FedUnlearn(base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    pass
    
def init_FedUnlearn_ctx(base_trainer):
    """
    init necessary attributes used in FedUnlearn,
    `global_model` acts as the shared global model in FedAvg;
    `local_model` acts as personalized model will be optimized with regularization based on weights of `global_model`;
    `loss_thresh` acts as the threshold for samples loss;
    `trap_rate` acts as the rate at which the low loss samples are treated as backdoor samples
    `switch_rounds` acts as the rounds number of when to switch to stage 2 (i.e., the unlearning stage).
    """
    pass