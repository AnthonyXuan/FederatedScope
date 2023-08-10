# 需要在criterion层面，对不同样本进行区分
# 需要增加一（两）个context变量，存放后门样本和干净样本
# 根据通信轮数进行方案两个阶段的区别

import copy
import logging
import torch
import numpy as np

from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from federatedscope.core.auxiliaries.dataloader_builder import WrapDataset
from typing import Type
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)

def wrap_FedUnlearnTrainer(base_trainer: Type[GeneralTorchTrainer]) -> Type[GeneralTorchTrainer]:
    init_FedUnlearn_ctx(base_trainer)

    # HOOK_TRIGGER = [
    #     "on_fit_start", "on_epoch_start", "on_batch_start", "on_batch_forward",
    #     "on_batch_backward", "on_batch_end", "on_epoch_end", "on_fit_end"
    # ]
    
    # ! replace the hook '_hook_on_batch_forward' with '_hook_on_batch_forward_fedunlearn'
    base_trainer.replace_hook_in_train(new_hook=_hook_on_batch_forward_fedunlearn, target_trigger="on_batch_forward", target_hook_name='_hook_on_batch_forward')
    base_trainer.replace_hook_in_train(new_hook=_hook_on_epoch_start_fedunlearn, target_trigger="on_epoch_start", target_hook_name='_hook_on_epoch_start')
    base_trainer.replace_hook_in_train(new_hook=_hook_on_batch_start_init_fedunlearn, target_trigger="on_batch_start", target_hook_name='_hook_on_batch_start_init')
    
    # ! modify this hook to be append, rather than replacement
    base_trainer.register_hook_in_train(new_hook=_hook_on_fit_end_devide_dataset,
                                        trigger='on_fit_end',
                                        insert_pos=0)

    # ! disable hooks for calculating flops
    base_trainer.reset_hook_in_train(target_trigger='on_batch_forward', target_hook_name='_hook_on_batch_forward_flop_count')




    base_trainer.register_hook_in_train(new_hook=_hook_on_fit_start_clean,
                                        trigger='on_fit_start',
                                        insert_pos=-1)
    base_trainer.register_hook_in_train(
        new_hook=_hook_on_fit_start_set_regularized_para,
        trigger="on_fit_start",
        insert_pos=0)
    base_trainer.register_hook_in_train(
        new_hook=_hook_on_batch_start_switch_model,
        trigger="on_batch_start",
        insert_pos=0)
    base_trainer.register_hook_in_train(
        new_hook=_hook_on_batch_forward_cnt_num,
        trigger="on_batch_forward",
        insert_pos=-1)
    base_trainer.register_hook_in_train(new_hook=_hook_on_batch_end_flop_count,
                                        trigger="on_batch_end",
                                        insert_pos=-1)
    # base_trainer.register_hook_in_train(new_hook=_hook_on_fit_end_calibrate,
    #                                     trigger='on_fit_end',
    #                                     insert_pos=-1)
    # evaluation is based on the local personalized model
    base_trainer.register_hook_in_eval(
        new_hook=_hook_on_fit_start_switch_local_model,
        trigger="on_fit_start",
        insert_pos=0)
    base_trainer.register_hook_in_eval(
        new_hook=_hook_on_fit_end_switch_global_model,
        trigger="on_fit_end",
        insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=_hook_on_fit_end_free_cuda,
                                        trigger="on_fit_end",
                                        insert_pos=-1)
    base_trainer.register_hook_in_eval(new_hook=_hook_on_fit_end_free_cuda,
                                       trigger="on_fit_end",
                                       insert_pos=-1)

    return base_trainer
    
# prepare the dataloader of current split
def _hook_on_epoch_start_fedunlearn(ctx):
    """
    Note:
        The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.{ctx.cur_data_split}_loader``      Initialize DataLoader
        ==================================  ===========================
    """
    # ! Since I only register hooks in train, "ctx.cur_data_split == 'train' " may be unnecessary.
    # ! Anthony mark
    # ! I didn't use wrapDataset here
    if ctx.world_state > ctx.switch_rounds:
        benign_loader = get_dataloader(
            ctx.get("benign_data"), ctx.cfg)
        setattr(ctx, "benign_loader", ReIterator(benign_loader))
        if len(ctx.get("backdoor_data")) != 0:
            backdoor_loader = get_dataloader(
                ctx.get("backdoor_data"), ctx.cfg)
            setattr(ctx, "backdoor_loader", ReIterator(backdoor_loader))
        else:
            setattr(ctx, "backdoor_loader", None)
    else:
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_data_split)) is None:
            loader = get_dataloader(
                WrapDataset(ctx.get("{}_data".format(ctx.cur_data_split))),
                ctx.cfg)
            setattr(ctx, "{}_loader".format(ctx.cur_data_split), ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_data_split)),
                            ReIterator):
            setattr(ctx, "{}_loader".format(ctx.cur_data_split),
                    ReIterator(ctx.get("{}_loader".format(ctx.cur_data_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_data_split)).reset()

# prepare the current batch data
def _hook_on_batch_start_init_fedunlearn(ctx):
    """
    Note:
        The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.data_batch``                  Initialize batch data
        ==================================  ===========================
    """
    
    if ctx.world_state > ctx.switch_rounds:
        # prepare data batch
        try:
            ctx.benign_data_batch = next(ctx.get("benign_loader"))
        except StopIteration:
            raise StopIteration
        
        if ctx.get("backdoor_loader") != None:
            try:
                ctx.backdoor_data_batch = next(ctx.get("backdoor_loader"))
            except StopIteration:
                raise StopIteration
        else:
            ctx.backdoor_data_batch = None
            
    else:
        # prepare data batch
        try:
            ctx.data_batch = next(ctx.get("{}_loader".format(ctx.cur_data_split)))
        except StopIteration:
            raise StopIteration
    
    
def _hook_on_batch_forward_fedunlearn(ctx):

    if ctx.world_state > ctx.switch_rounds:
        if ctx.backdoor_data_batch != None:
            x, label_benign = [_.to(ctx.device) for _ in ctx.benign_data_batch]
            pred_benign = ctx.model(x)
            if len(label_benign.size()) == 0:
                label_benign = label_benign.unsqueeze(0)

            x, label_backdoor = [_.to(ctx.device) for _ in ctx.backdoor_data_batch]
            pred_backdoor = ctx.model(x)
            if len(label_backdoor.size()) == 0:
                label_backdoor = label_backdoor.unsqueeze(0)

            pred = torch.cat((pred_benign, pred_backdoor))
            label = torch.cat((label_benign, label_backdoor))
        
            ctx.y_true = label
            ctx.y_prob = pred
            # treat benign and backdoor data differently
            ctx.loss_batch = ctx.criterion(pred_benign, label_benign) - ctx.criterion(pred_backdoor, label_backdoor)
            ctx.batch_size = len(label)
        else:
            x, label_benign = [_.to(ctx.device) for _ in ctx.benign_data_batch]
            pred_benign = ctx.model(x)
            if len(label_benign.size()) == 0:
                label_benign = label_benign.unsqueeze(0)

            pred = pred_benign
            label = label_benign
        
            ctx.y_true = label
            ctx.y_prob = pred

            ctx.loss_batch = ctx.criterion(pred_benign, label_benign)
            ctx.batch_size = len(label)

    else:
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = label
        ctx.y_prob = pred
        
        default_reduction = getattr(ctx.criterion, 'reduction')
        setattr(ctx.criterion, 'reduction', 'none')
        loss_per_sample = ctx.criterion(pred, label)
        setattr(ctx.criterion, 'reduction', default_reduction)
        
        loss_batch_value = torch.sum(
            torch.sign(loss_per_sample - ctx.loss_thresh) * loss_per_sample
        ) 
        
        ctx.loss_batch = loss_batch_value
        ctx.batch_size = len(label)
    
# ! modify this function to classify benign data and backdoor data
def _hook_on_fit_end_devide_dataset(ctx):

    if ctx.world_state == ctx.switch_rounds:
        # save default reduction parameters
        default_reduction = getattr(ctx.criterion, 'reduction')
        # change reduction parameters to none reduction
        setattr(ctx.criterion, 'reduction', 'none')
        # create data_loader for train data
        data_loader= DataLoader(ctx.data['train'].dataset, batch_size=ctx.cfg.data.batch_size, shuffle=False, num_workers=ctx.cfg.data.num_workers)
        # init loss_all_sample
        loss_all_sample = torch.empty(0).to(ctx.device)
        for batch_data, batch_label in data_loader:
            batch_data = batch_data.to(ctx.device)
            batch_label = batch_label.to(ctx.device)
            
            batch_pred = ctx.model(batch_data)
            loss_per_sample = ctx.criterion(batch_pred, batch_label)
            
            loss_all_sample = torch.cat((loss_all_sample, loss_per_sample))
            
        # restore default reduction parameters
        setattr(ctx.criterion, 'reduction', default_reduction)
        # judge which samples should be classified as backdoored
        top_k = round(ctx.trap_rate * len(ctx.data['train'].dataset))
        topk_val, topk_indx = torch.topk(loss_all_sample, top_k, largest=False, sorted=True)
        topk_indx_final = []
        topk_flag = topk_val < ctx.loss_thresh
        for i, flag in enumerate(topk_flag):
            if flag:
                topk_indx_final.append(topk_indx[i])
        topk_indx_final = torch.stack(topk_indx_final)
        
        train_data = [item for item in ctx.data['train'].dataset]
        backdoor_data = [train_data[indx] for indx in topk_indx_final]
        mask = torch.ones([len(train_data)], dtype=torch.bool)
        # mask = torch.ones_like(train_data, dtype=torch.bool)
        mask[topk_indx_final] = False
        
        benign_data = [train_data[i] for i, flag in enumerate(mask) if flag]
        
        setattr(ctx, 'benign_data', benign_data)
        setattr(ctx, 'backdoor_data', backdoor_data)
    
def init_FedUnlearn_ctx(base_trainer):
    """
    init necessary attributes used in FedUnlearn,
    `global_model` acts as the shared global model in FedAvg;
    `local_model` acts as personalized model will be optimized with regularization based on weights of `global_model`;
    `loss_thresh` acts as the threshold for samples loss;
    `trap_rate` acts as the rate at which the low loss samples are treated as backdoor samples
    `switch_rounds` acts as the rounds number of when to switch to stage 2 (i.e., the unlearning stage).
    """
    
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg
    
    # used in WrapDataset()
    ctx.cfg = cfg
    ctx.global_model = copy.deepcopy(ctx.model)
    ctx.local_model = copy.deepcopy(ctx.model)
    ctx.loss_thresh = cfg.fedunlearn.loss_thresh
    ctx.trap_rate = cfg.fedunlearn.trap_rate
    ctx.switch_rounds = cfg.fedunlearn.switch_rounds
    
    # ! Very important for federated learning, because most clients won't get involved in stage 1
    # default benign data and backdoor data
    ctx.benign_data = [item for item in ctx.data['train'].dataset]
    ctx.backdoor_data = []
    
    ctx.models = [ctx.local_model, ctx.global_model]
    
    ctx.model = ctx.global_model
    ctx.use_local_model_current = False
    
    ctx.num_samples_local_model_train = 0
    
    ctx.num_train_batch_for_local_model, \
        ctx.num_train_batch_last_epoch_for_local_model, \
        ctx.num_train_epoch_for_local_model, \
        ctx.num_total_train_batch \
        = ctx.pre_calculate_batch_epoch_num \
        (cfg.personalization.local_update_steps)

    if cfg.federate.batch_or_epoch == 'batch':
        ctx.num_train_batch += ctx.num_train_batch_for_local_model
        ctx.num_train_batch_last_epoch += \
            ctx.num_train_batch_last_epoch_for_local_model
    else:
        ctx.num_train_epoch += ctx.num_train_epoch_for_local_model
    
    
# set optimizer for regularized model, i.e. the local model. And set optimizer for global model.
def _hook_on_fit_start_set_regularized_para(ctx):

    # set the compared model data for local personalized model
    ctx.global_model.to(ctx.device)
    ctx.local_model.to(ctx.device)
    ctx.global_model.train()
    ctx.local_model.train()
    compared_global_model_para = [{
        "params": list(ctx.global_model.parameters())
    }]

    ctx.optimizer_for_global_model = get_optimizer(ctx.global_model,
                                                   **ctx.cfg.optimizer)
    ctx.optimizer_for_local_model = get_optimizer(ctx.local_model,
                                                  **ctx.cfg.optimizer)

    ctx.optimizer_for_local_model = wrap_regularized_optimizer(
        ctx.optimizer_for_local_model, ctx.cfg.personalization.regular_weight)

    ctx.optimizer_for_local_model.set_compared_para_group(
        compared_global_model_para)

# clean the default optimizer, named 'ctx.optimizer'
def _hook_on_fit_start_clean(ctx):

    # remove the unnecessary optimizer
    del ctx.optimizer
    ctx.num_samples_local_model_train = 0

# ! in backdoor branch, this hook function is not used
# def _hook_on_fit_end_calibrate(ctx):

#     # make the num_samples_train only related to the global model.
#     # (num_samples_train will be used in aggregation process)
#     ctx.num_samples -= ctx.num_samples_local_model_train
#     ctx.eval_metrics['train_total'] = ctx.num_samples
#     ctx.eval_metrics['train_total_local_model'] = \
#         ctx.num_samples_local_model_train

# calculate the the flops during current training batch
def _hook_on_batch_end_flop_count(ctx):

    # besides the normal forward flops, the regularization adds the cost of
    # number of model parameters
    ctx.monitor.total_flops += ctx.monitor.total_model_size / 2

# count the number of samples in the whole local model trainning process
def _hook_on_batch_forward_cnt_num(ctx):

    if ctx.use_local_model_current:
        ctx.num_samples_local_model_train += ctx.batch_size

# swtich the model(between global and local model) and the optimizer for training
def _hook_on_batch_start_switch_model(ctx):

    if ctx.cfg.federate.batch_or_epoch == 'batch':
        if ctx.cur_epoch_i == (ctx.num_train_epoch - 1):
            ctx.use_local_model_current = \
                ctx.cur_batch_i < \
                ctx.num_train_batch_last_epoch_for_local_model
        else:
            ctx.use_local_model_current = \
                ctx.cur_batch_i < ctx.num_train_batch_for_local_model
    else:
        ctx.use_local_model_current = \
            ctx.cur_epoch_i < ctx.num_train_epoch_for_local_model

    # switch model to change ctx that used in default hooks
    if ctx.use_local_model_current:
        ctx.model = ctx.local_model
        ctx.optimizer = ctx.optimizer_for_local_model
    else:
        ctx.model = ctx.global_model
        ctx.optimizer = ctx.optimizer_for_global_model

# use local_model before each fitting
def _hook_on_fit_start_switch_local_model(ctx):

    ctx.model = ctx.local_model
    ctx.model.eval()

# use global_model after each fitting
def _hook_on_fit_end_switch_global_model(ctx):

    ctx.model = ctx.global_model


def _hook_on_fit_end_free_cuda(ctx):

    ctx.global_model.to(torch.device("cpu"))
    ctx.local_model.to(torch.device("cpu"))