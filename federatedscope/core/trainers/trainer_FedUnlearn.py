# 需要在criterion层面，对不同样本进行区分
# 需要增加一（两）个context变量，存放后门样本和干净样本
# 根据通信轮数进行方案两个阶段的区别

import copy
import logging
import torch
import numpy as np
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.core.auxiliaries.ReIterator import ReIterator

from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from federatedscope.core.trainers.utils import calculate_batch_epoch_num
from federatedscope.core.data.wrap_dataset import WrapDataset
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
    base_trainer.register_hook_in_train(new_hook=_hook_on_fit_end_calibrate,
                                        trigger='on_fit_end',
                                        insert_pos=-1)
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
        ``ctx.{ctx.cur_split}_loader``      Initialize DataLoader
        ==================================  ===========================
    """
    # ! Since I only register hooks in train, "ctx.cur_split == 'train' " may be unnecessary.
    if ctx.world_state > ctx.switch_rounds:
        benign_loader = get_dataloader(
            WrapDataset(ctx.get("benign_data")), ctx.cfg, 'train')
        setattr(ctx, "benign_loader", ReIterator(benign_loader))
        backdoor_loader = get_dataloader(
            WrapDataset(ctx.get("backdoor_data")), ctx.cfg, 'train')
        setattr(ctx, "backdoor_loader", ReIterator(backdoor_loader))
    else:
        # prepare dataloader
        if ctx.get("{}_loader".format(ctx.cur_split)) is None:
            loader = get_dataloader(
                WrapDataset(ctx.get("{}_data".format(ctx.cur_split))),
                ctx.cfg, ctx.cur_split)
            setattr(ctx, "{}_loader".format(ctx.cur_split), ReIterator(loader))
        elif not isinstance(ctx.get("{}_loader".format(ctx.cur_split)),
                            ReIterator):
            setattr(ctx, "{}_loader".format(ctx.cur_split),
                    ReIterator(ctx.get("{}_loader".format(ctx.cur_split))))
        else:
            ctx.get("{}_loader".format(ctx.cur_split)).reset()

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
            ctx.benign_data_batch = CtxVar(
                next(ctx.get("benign_loader")),
                LIFECYCLE.BATCH)
        except StopIteration:
            raise StopIteration
        
        try:
            ctx.backdoor_data_batch = CtxVar(
                next(ctx.get("backdoor_loader")),
                LIFECYCLE.BATCH)
        except StopIteration:
            raise StopIteration
    else:
        # prepare data batch
        try:
            ctx.data_batch = CtxVar(
                next(ctx.get("{}_loader".format(ctx.cur_split))),
                LIFECYCLE.BATCH)
        except StopIteration:
            raise StopIteration
    
    
def _hook_on_batch_forward_fedunlearn(ctx):
    """
    Note:
        The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.y_true``                      Move to `ctx.device`
        ``ctx.y_prob``                      Forward propagation get y_prob
        ``ctx.loss_batch``                  Calculate the loss
        ``ctx.batch_size``                  Get the batch_size
        ==================================  ===========================
    """
    if ctx.world_state > ctx.switch_rounds:
        
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
    
        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        # treat benign and backdoor data differently
        ctx.loss_batch = CtxVar(ctx.criterion(pred_benign, label_benign) - ctx.criterion(pred_backdoor, label_backdoor), LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
    # TODO: 最后一个round，划分benign和backdoor数据集
    # TODO： 需要解决问题： 如何通过ctx变量定位当前epoch，batch和全部所需epoch，batch
    else:
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.y_true = CtxVar(label, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        
        default_reduction = getattr(ctx.criterion, 'reduction')
        setattr(ctx.criterion, 'reduction', 'none')
        loss_per_sample = ctx.criterion(pred, label)
        setattr(ctx.criterion, 'reduction', default_reduction)
        
        loss_batch_value = torch.sum(
            torch.sign(loss_per_sample - ctx.loss_thresh) * loss_per_sample
        ) 
        
        ctx.loss_batch = CtxVar(loss_batch_value, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(label), LIFECYCLE.BATCH)
    
# ! modify this function to classify benign data and backdoor data
def _hook_on_fit_end_fedunlearn(ctx):
    """
    Evaluate metrics.

    Note:
        The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.ys_true``                     Convert to ``numpy.array``
        ``ctx.ys_prob``                     Convert to ``numpy.array``
        ``ctx.monitor``                     Evaluate the results
        ``ctx.eval_metrics``                Get evaluated results from \
        ``ctx.monitor``
        ==================================  ===========================
    """
    if ctx.world_state == ctx.switch_rounds:
        # save default reduction parameters
        default_reduction = getattr(ctx.criterion, 'reduction')
        # change reduction parameters to none reduction
        setattr(ctx.criterion, 'reduction', 'none')
        # create data_loader for train data
        data_loader= DataLoader(WrapDataset(ctx.train_data), batch_size=ctx.dataloader.batch_size, shuffle=False, num_workers=ctx.dataloader.num_workers)
        # init loss_all_sample
        loss_all_sample = torch.empty(0)
        for batch_data, batch_label in data_loader:
            batch_data = batch_data.to(ctx.device)
            batch_label = batch_label.to(ctx.device)
            
            batch_pred = ctx.model(batch_data)
            loss_per_sample = ctx.criterion(batch_pred, batch_label)
            
            loss_all_sample = torch.cat((loss_all_sample, loss_per_sample))
            
        # restore default reduction parameters
        setattr(ctx.criterion, 'reduction', default_reduction)
        # judge which samples should be classified as backdoored
        top_k = round(ctx.trap_rate * len(ctx.train_data))
        topk_val, topk_indx = torch.topk(loss_all_sample, top_k, largest=False, sorted=True)
        topk_indx_final = []
        topk_flag = topk_val < ctx.loss_thresh
        for i, flag in enumerate(topk_flag):
            if flag:
                topk_indx_final.append(topk_indx[i])
        
        benign_data = ctx.train_data[topk_indx_final]
        mask = torch.ones_like(ctx.train_data, dtype=torch.bool)
        mask[topk_indx_final] = False
        backdoor_data = ctx.train_data[mask]
        
        setattr(ctx, 'benign_data', benign_data)
        setattr(ctx, 'backdoor_data', backdoor_data)
        

    ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
    ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
    results = ctx.monitor.eval(ctx)
    setattr(ctx, 'eval_metrics', results)
    
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
    
    ctx.benign_data = None
    ctx.backdoor_data = None
    
    ctx.models = [ctx.local_model, ctx.global_model]
    
    ctx.model = ctx.global_model
    ctx.use_local_model_current = False
    
    ctx.num_samples_local_model_train = 0
    
    # track the batch_num, epoch_num, for local & global model respectively
    cfg_p_local_update_steps = cfg.personalization.local_update_steps
    ctx.num_train_batch_for_local_model, \
        ctx.num_train_batch_last_epoch_for_local_model, \
        ctx.num_train_epoch_for_local_model, \
        ctx.num_total_train_batch = \
        calculate_batch_epoch_num(cfg_p_local_update_steps,
                                  cfg.train.batch_or_epoch,
                                  ctx.num_train_data,
                                  cfg.dataloader.batch_size,
                                  cfg.dataloader.drop_last)

    # In the first
    # 1. `num_train_batch` and `num_train_batch_last_epoch`
    # (batch_or_epoch == 'batch' case) or
    # 2. `num_train_epoch`,
    # (batch_or_epoch == 'epoch' case)
    # we will manipulate local models, and manipulate global model in the
    # remaining steps
    if cfg.train.batch_or_epoch == 'batch':
        ctx.num_train_batch += ctx.num_train_batch_for_local_model
        ctx.num_train_batch_last_epoch += \
            ctx.num_train_batch_last_epoch_for_local_model
    else:
        ctx.num_train_epoch += ctx.num_train_epoch_for_local_model
    
    
# set optimizer for regularized model, i.e. the local model. And set optimizer for global model.
def _hook_on_fit_start_set_regularized_para(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.global_model``                Move to ``ctx.device`` and set \
        to ``train`` mode
        ``ctx.local_model``                 Move to ``ctx.device`` and set \
        to ``train`` mode
        ``ctx.optimizer_for_global_model``  Initialize by ``ctx.cfg`` and \
        wrapped by ``wrap_regularized_optimizer``
        ``ctx.optimizer_for_local_model``   Initialize by ``ctx.cfg`` and \
        set compared parameter group
        ==================================  ===========================
    """
    # set the compared model data for local personalized model
    ctx.global_model.to(ctx.device)
    ctx.local_model.to(ctx.device)
    ctx.global_model.train()
    ctx.local_model.train()
    compared_global_model_para = [{
        "params": list(ctx.global_model.parameters())
    }]

    ctx.optimizer_for_global_model = get_optimizer(ctx.global_model,
                                                   **ctx.cfg.train.optimizer)
    ctx.optimizer_for_local_model = get_optimizer(ctx.local_model,
                                                  **ctx.cfg.train.optimizer)

    ctx.optimizer_for_local_model = wrap_regularized_optimizer(
        ctx.optimizer_for_local_model, ctx.cfg.personalization.regular_weight)

    ctx.optimizer_for_local_model.set_compared_para_group(
        compared_global_model_para)

# clean the default optimizer, named 'ctx.optimizer'
def _hook_on_fit_start_clean(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.optimizer``                   Delete
        ``ctx.num_..._local_model_train``   Initialize to 0
        ==================================  ===========================
    """
    # remove the unnecessary optimizer
    del ctx.optimizer
    ctx.num_samples_local_model_train = 0

# 
def _hook_on_fit_end_calibrate(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.num_samples``                 Minus \
        ``ctx.num_samples_local_model_train``
        ``ctx.eval_metrics``                Record ``train_total`` and \
        ``train_total_local_model``
        ==================================  ===========================
    """
    # make the num_samples_train only related to the global model.
    # (num_samples_train will be used in aggregation process)
    ctx.num_samples -= ctx.num_samples_local_model_train
    ctx.eval_metrics['train_total'] = ctx.num_samples
    ctx.eval_metrics['train_total_local_model'] = \
        ctx.num_samples_local_model_train

# calculate the the flops during current training batch
def _hook_on_batch_end_flop_count(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.monitor``                     Monitor total flops
        ==================================  ===========================
    """
    # besides the normal forward flops, the regularization adds the cost of
    # number of model parameters
    ctx.monitor.total_flops += ctx.monitor.total_model_size / 2

# count the number of samples in the whole local model trainning process
def _hook_on_batch_forward_cnt_num(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.num_..._local_model_train``   Add `ctx.batch_size`
        ==================================  ===========================
    """
    if ctx.use_local_model_current:
        ctx.num_samples_local_model_train += ctx.batch_size

# swtich the model(between global and local model) and the optimizer for training
def _hook_on_batch_start_switch_model(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.use_local_model_current``     Set to ``True`` or ``False``
        ``ctx.model``                       Set to ``ctx.local_model`` or \
        ``ctx.global_model``
        ``ctx.optimizer``                   Set to \
        ``ctx.optimizer_for_local_model`` or ``ctx.optimizer_for_global_model``
        ==================================  ===========================
    """
    if ctx.cfg.train.batch_or_epoch == 'batch':
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


# Note that Ditto only updates the para of global_model received from other
# FL participants, and in the remaining steps, ctx.model has been =
# ctx.global_model, thus we do not need register the following hook
# def hook_on_fit_end_link_global_model(ctx):
#     ctx.model = ctx.global_model

# use local_model before each fitting
def _hook_on_fit_start_switch_local_model(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.model``                       Set to ``ctx.local_model`` and \
        set to ``eval`` mode
        ==================================  ===========================
    """
    ctx.model = ctx.local_model
    ctx.model.eval()

# use global_model after each fitting
def _hook_on_fit_end_switch_global_model(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.model ``                      Set to ``ctx.global_model``
        ==================================  ===========================
    """
    ctx.model = ctx.global_model


def _hook_on_fit_end_free_cuda(ctx):
    """
    Note:
      The modified attributes and according operations are shown below:
        ==================================  ===========================
        Attribute                           Operation
        ==================================  ===========================
        ``ctx.global_model``                Move to ``cpu``
        ``ctx.locol_model``                 Move to ``cpu``
        ==================================  ===========================
    """
    ctx.global_model.to(torch.device("cpu"))
    ctx.local_model.to(torch.device("cpu"))