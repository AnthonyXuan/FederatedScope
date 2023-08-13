import yaml
import os
# base_folder = './new-scripts'
# base_folder = './new-scripts-high-attack'
base_folder = './200-rounds-scripts'
out_dir = './200-rounds-output'

class BaseConfig:
    def __init__(self):
        # fixed sample rate, total rounds, client number.
        self.use_gpu = True
        self.early_stop = {"patience": 0}
        self.seed = 12345
        self.federate = {
            "mode": "standalone",
            "client_num": 100,
            "total_round_num": 200,
            "sample_client_rate": 0.1,
            "make_global_eval": False,
            "batch_or_epoch": "epoch"
        }
        # fixed dataset, and lda splitter alpha to be 0.5
        self.data = {
            "root": "data/",
            "type": "CIFAR10@torchvision",
            # "splits": [0.8, 0.1, 0.1],
            "splits": [1.0, 0.0, 0.0],
            "num_workers": 0,
            "transform": [['ToTensor']],
            # "transform": [["ToTensor"], ["Normalize", {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]}]],
            "args": [{"download": True}],
            "splitter": "lda",
            "splitter_args": [{"alpha": 0.5}],
            "batch_size": 64,
            # ! 'dataset' is new in backdoor branch 
            "dataset": ['train', 'val', 'test', 'poison']
        }
        # self.dataloader = {
        #     "batch_size": 64
        # }
        self.model = {
            "type": "convnet2",
            "hidden": 512,
            "out_channels": 10,
            "dropout": 0.0
        }
        self.optimizer = {
            "weight_decay": 0.0
        }
        self.grad = {
            "grad_clip": 5.0
        }
        self.criterion = {
            "type": "CrossEntropyLoss"
        }
        self.trainer = {
            "type": "cvtrainer"
        }
        self.eval = {
            "freq": 100,
            "metrics": ["acc", "correct"],
            "best_res_update_round_wise_key": "test_loss",
            # "split": ["test", "val"],
            "split": ['test','poison'],
            # ! 'metrics' is new in backdoor branch
            "metrics": ['acc', 'correct']
        }
        self.outdir = out_dir
        self.verbose = 1

class Attack():
    def __init__(self, attack_type="naive",use_multi_attackers=False, attackers_list=[], attacker_id=1, attack_settings='fix', poison_rate=0.05):
        if attack_type not in ['naive', 'badnet', 'narci', 'hk', 'signal']:
            print('Abort! Bad attack type!')
        if attack_type == 'naive':
            self.attack = {
                "attacker_id": -1,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            }
            return
        elif attack_type == 'badnet':
            self.attack = {
                "attack_method": "backdoor",
                "poison_ratio": 0.05,
                "freq": 3,
                "trigger_type": "squareTrigger",
                "label_type": "dirty",
                "attacker_id": 1,
                "target_label_ind": 7,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            }
        elif attack_type == 'narci':
            self.attack = {
                "attack_method": "backdoor",
                "poison_ratio": 0.05,
                "freq": 3,
                "trigger_type": "narciTrigger",
                "label_type": "clean",
                "attacker_id": 1,
                "target_label_ind": 2,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            }
        elif attack_type == 'hk':
            self.attack = {
                "attack_method": "backdoor",
                "poison_ratio": 0.05,
                "freq": 3,
                "trigger_type": "hkTrigger",
                "label_type": "dirty",
                "attacker_id": 1,
                "target_label_ind": 7,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            }
        elif attack_type == 'signal':
            self.attack = {
                "attack_method": "backdoor",
                "poison_ratio": 0.05,
                "freq": 3,
                "trigger_type": "signalTrigger",
                "label_type": "dirty",
                "attacker_id": 1,
                "target_label_ind": 7,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            }
            
        self.attack['use_multi_attackers'] = use_multi_attackers
        self.attack['attackers_list'] = attackers_list
        self.attack['attacker_id'] = attacker_id
        self.attack['setting'] = attack_settings
        
class MyAttack(Attack):
    def __init__(self, attack_type="naive", multi_attack=False):
        if multi_attack:
            super().__init__(attack_type, use_multi_attackers=True, attackers_list=list(range(1,6)), attack_settings='fix', poison_rate=0.01)
        else:
            super().__init__(attack_type, use_multi_attackers=False)

class FedAvgConfig(BaseConfig, MyAttack):
    def __init__(self, attack_type, use_multiple_attackers):
        BaseConfig.__init__(self)
        MyAttack.__init__(self, attack_type=attack_type, multi_attack=use_multiple_attackers)
        self.federate["method"] = "FedAvg"
        self.optimizer["lr"] = 0.1
        self.federate["local_update_steps"] = 2
        self.expname =  attack_type +  '_fedavg'
        self.device = 0
        
# ! It seems this algo sucks
class FedUnlearnConfig(BaseConfig, MyAttack):
    def __init__(self, attack_type, use_multiple_attackers):
        BaseConfig.__init__(self)
        MyAttack.__init__(self, attack_type=attack_type, multi_attack=use_multiple_attackers)
        self.federate["method"] = "FedUnlearn"
        self.federate["local_update_steps"] = 2
        self.optimizer["lr"] = 0.1
        self.expname =  attack_type +  '_fedunlearn'
        self.device = 0
        self.fedunlearn = {
            'loss_thresh': 0.5,
            ' _rate': 0.05,
            'switch_rounds': 9
        }
        
class DittoConfig(BaseConfig, MyAttack):
    def __init__(self, attack_type, use_multiple_attackers):
        BaseConfig.__init__(self)
        MyAttack.__init__(self, attack_type=attack_type, multi_attack=use_multiple_attackers)
        self.federate["method"] = "Ditto"
        self.optimizer["lr"] = 0.1
        self.federate["local_update_steps"] = 2
        self.expname =  attack_type +  '_ditto'
        self.device = 1
        
class pFedMeConfig(BaseConfig, MyAttack):
    def __init__(self, attack_type, use_multiple_attackers):
        BaseConfig.__init__(self)
        MyAttack.__init__(self, attack_type=attack_type, multi_attack=use_multiple_attackers)
        self.federate["method"] = "pFedMe"
        self.optimizer["lr"] = 0.1
        self.federate["local_update_steps"] = 3
        self.personalization = {
            'regular_weight' : 0.5,
            'K': 2,
            'beta': 1.0
        }
        self.expname =  attack_type +  '_pfedme'
        self.device = 2
        
class FedRepConfig(BaseConfig, MyAttack):
    def __init__(self, attack_type, use_multiple_attackers):
        BaseConfig.__init__(self)
        MyAttack.__init__(self, attack_type=attack_type, multi_attack=use_multiple_attackers)
        self.federate["method"] = "FedRep"
        self.optimizer["lr"] = 0.1
        self.federate["local_update_steps"] = 2
        self.personalization = {
            'lr_feature' : 0.1,
            'lr_linear': 0.005,
            'epoch_feature': 1,
            'epoch_linear': 1,
            'local_param': ["fc2"],
            'share_non_trainable_para': False
        }
        self.expname =  attack_type +  '_fedrep'
        self.device = 3
        

def write_config_to_yaml(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=None)

cfgs = []

attack_types = ['naive', 'badnet', 'narci', 'hk', 'signal']
ditto_config = DittoConfig
fedavg_config = FedAvgConfig
# fedunlearn_config = FedUnlearnConfig
pfedme_config = pFedMeConfig
fedrep_config = FedRepConfig

is_multiple = False

for attack_type in attack_types:
    cfgs.append(ditto_config(attack_type=attack_type, use_multiple_attackers=is_multiple))
    cfgs.append(fedavg_config(attack_type=attack_type, use_multiple_attackers=is_multiple))
    # cfgs.append(fedunlearn_config(attack_type=attack_type, use_multiple_attackers=is_multiple))
    cfgs.append(pfedme_config(attack_type=attack_type, use_multiple_attackers=is_multiple))
    cfgs.append(fedrep_config(attack_type=attack_type, use_multiple_attackers=is_multiple))

for cfg in cfgs:
    write_config_to_yaml(cfg, os.path.join(base_folder, cfg.expname + '.yaml'))