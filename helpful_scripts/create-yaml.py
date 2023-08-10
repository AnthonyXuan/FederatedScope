import yaml
import os
# base_folder = './new-scripts'
base_folder = './new-scripts-high-attack'

class BaseConfig:
    def __init__(self):
        # fixed sample rate, total rounds, client number.
        self.use_gpu = True
        self.early_stop = {"patience": 0}
        self.seed = 12345
        self.federate = {
            "mode": "standalone",
            "client_num": 100,
            "total_round_num": 500,
            "sample_client_rate": 0.1,
            "make_global_eval": False,
            "batch_or_epoch": "epoch"
        }
        # fixed dataset, and lda splitter alpha to be 0.5
        self.data = {
            "root": "data/",
            "type": "CIFAR10@torchvision",
            "splits": [0.8, 0.1, 0.1],
            "num_workers": 0,
            "transform": [["ToTensor"], ["Normalize", {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]}]],
            "args": [{"download": True}],
            "splitter": "lda",
            "splitter_args": [{"alpha": 0.5}],
            "batch_size": 64
        }
        # self.dataloader = {
        #     "batch_size": 64
        # }
        self.model = {
            "type": "convnet2",
            "hidden": 2048,
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
            "freq": 1,
            "metrics": ["acc", "correct"],
            "best_res_update_round_wise_key": "test_loss",
            "split": ["test", "val"]
        }
        self.outdir = 'new-output/'
        self.verbose = 1

class Attack():
    def __init__(self, attack_type="naive"):
        if attack_type not in ['naive', 'badnet', 'narci']:
            print('Abort! Bad attack type!')
        if attack_type == 'naive':
            self.attack = {
                "attack_method": "backdoor",
                "setting": "fix",
                "poison_ratio": 0.00001,
                "freq": 100,
                "trigger_type": "squareTrigger",
                "label_type": "dirty",
                "attacker_id": 1,
                "target_label_ind": 7,
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            }
        elif attack_type == 'badnet':
            self.attack = {
                "attack_method": "backdoor",
                "setting": "fix",
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
                "setting": "fix",
                "poison_ratio": 0.05,
                "freq": 3,
                "trigger_type": "signalTrigger",
                "label_type": "clean",
                "attacker_id": 4,
                "target_label_ind": 2,
                "trigger_path": "my_trigger/",
                "mean": [0.4914, 0.4822, 0.4465],
                "std": [0.2470, 0.2435, 0.2616]
            }

class FedUnlearnConfig(BaseConfig, Attack):
    def __init__(self, attack_type):
        BaseConfig.__init__(self)
        Attack.__init__(self, attack_type=attack_type)
        self.federate["method"] = "FedUnlearn"
        self.federate["local_update_steps"] = 2
        self.optimizer["lr"] = 0.1
        self.expname =  attack_type +  '_fedunlearn'
        self.device = 0
        self.fedunlearn = {
            'loss_thresh': 0.5,
            'trap_rate': 0.05,
            'switch_rounds': 9
        }

class DittoConfig(BaseConfig, Attack):
    def __init__(self, attack_type):
        BaseConfig.__init__(self)
        Attack.__init__(self, attack_type=attack_type)
        self.federate["method"] = "Ditto"
        self.optimizer["lr"] = 0.1
        self.federate["local_update_steps"] = 2
        self.expname =  attack_type +  '_ditto'
        self.device = 1

class FedAvgConfig(BaseConfig, Attack):
    def __init__(self, attack_type):
        BaseConfig.__init__(self)
        Attack.__init__(self, attack_type=attack_type)
        self.federate["method"] = "FedAvg"
        self.optimizer["lr"] = 0.1
        self.federate["local_update_steps"] = 2
        self.expname =  attack_type +  '_fedavg'
        self.device = 1

def write_config_to_yaml(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=None)

cfgs = []

attack_types = ['naive', 'badnet', 'narci']
ditto_config = DittoConfig
fedavg_config = FedAvgConfig
fedunlearn_config = FedUnlearnConfig

for attack_type in attack_types:
    cfgs.append(ditto_config(attack_type=attack_type))
    cfgs.append(fedavg_config(attack_type=attack_type))
    cfgs.append(fedunlearn_config(attack_type=attack_type))

for cfg in cfgs:
    write_config_to_yaml(cfg, os.path.join(base_folder, cfg.expname + '.yaml'))