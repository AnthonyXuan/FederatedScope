import yaml
import os

base_folder = './100-scripts-poison-2'
out_dir = './100-out-poison-2'

is_multiple = False

class BaseConfig:
    def __init__(self):
        self.use_gpu = True
        self.early_stop = {"patience": 0}
        self.seed = 12345
        self.federate = {
            "mode": "standalone",
            "method": "Ditto",
            "local_update_steps": 2,
            "batch_or_epoch": "epoch",
            "total_round_num": 100,
            "sample_client_rate": 0.1,
            "client_num": 100
        }
        self.personalization = {
            "K": 5,
            "beta": 1.0,
            "local_param": [],
            "local_update_steps": 2,
            "lr": 0.1,
            "regular_weight": 0.1,
            "share_non_trainable_para": False
        }
        self.data = {
            "seed": 1234,
            "dataset": ['train', 'val', 'test', 'poison'],
            "root": "data/",
            "type": "CIFAR10@torchvision",
            "splits": [1.0, 0.0, 0.0],
            "batch_size": 32,
            "num_workers": 0,
            "transform": [['ToTensor']],
            "args": [{"download": True}],
            "splitter": "lda",
            "splitter_args": [{"alpha": 0.5}]
        }
        self.model = {
            "type": "convnet2",
            "hidden": 512,
            "out_channels": 10,
            "dropout": 0.0
        }
        self.optimizer = {
            "lr": 0.1,
            "weight_decay": 0.0
        }
        self.grad = {"grad_clip": 5.0}
        self.criterion = {"type": "CrossEntropyLoss"}
        self.trainer = {"type": "cvtrainer"}
        self.eval = {
            "best_res_update_round_wise_key": 'test_loss',
            "freq": 1,
            "split": ['test','poison'],
            "metrics": ['acc', 'correct']
        }
        self.attack = {
            "setting": 'fix',
            "freq": 3,
            "attack_method": 'backdoor',
            "attacker_id": 16,
            "label_type": 'dirty',
            "trigger_type": "gridTrigger",
            "edge_num": 500,
            "poison_ratio": 0.5,
            "target_label_ind": 9,
            "self_opt": False,
            "self_lr": 0.1,
            "self_epoch": 6,
            "scale_poisoning": False,
            "scale_para": 3.0,
            "pgd_poisoning": False,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010]
        }
        self.expname = "bench_ditto_grid_bb_savemodel"

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
        self.attack['poison_ratio'] = poison_rate
        
        
class MyAttack(Attack):
    def __init__(self, poison_rate, attack_type="naive", multi_attack=False):
        if multi_attack:
            super().__init__(attack_type, use_multi_attackers=True, attackers_list=list(range(1,21)), attack_settings='random', poison_rate=0.02)
        else:
            super().__init__(attack_type, use_multi_attackers=False, poison_rate=poison_rate)
        
class DittoConfig(BaseConfig, MyAttack):
    def __init__(self, attack_type, use_multiple_attackers, poison_rate):
        BaseConfig.__init__(self)
        MyAttack.__init__(self, attack_type=attack_type, multi_attack=use_multiple_attackers, poison_rate=poison_rate)
        self.federate["method"] = "Ditto"
        self.optimizer["lr"] = 0.1
        self.federate["local_update_steps"] = 2
        self.expname =  attack_type +  f'_{poison_rate}' 
        self.device = 1
        

def write_config_to_yaml(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=None)

cfgs = []

attack_types = ['naive', 'badnet', 'narci', 'hk', 'signal']
poison_rate_list = [0.05, 0.1, 0.3, 0.5]

ditto_config = DittoConfig

for i in range(len(poison_rate_list)):
    for attack_type in attack_types:
        poison_rate = poison_rate_list[i]
        d = ditto_config(attack_type=attack_type, use_multiple_attackers=is_multiple, poison_rate=poison_rate)
        d.device = i
        cfgs.append(d)

if not os.path.exists(base_folder):
    os.makedirs(base_folder)

for cfg in cfgs:
    write_config_to_yaml(cfg, os.path.join(base_folder, cfg.expname + '.yaml'))