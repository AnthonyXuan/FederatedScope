# FOLDER_PATH="new-scripts-high-attack"
# FOLDER_PATH="200-rounds-scripts" 
# FOLDER_PATH="200-multiattack-scripts-2-20"
# FOLDER_PATH="200-multiattack-scripts-2-20-resnet"
FOLDER_PATH="200-single-scripts-resnet"

# naive
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/naive_ditto.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/naive_fedavg.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/naive_fedrep.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/naive_pfedme.yaml" &
# badnet
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_ditto.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_fedavg.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_fedrep.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_pfedme.yaml" &
# hk
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_ditto.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_fedavg.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_fedrep.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_pfedme.yaml" &
# signal
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_ditto.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_fedavg.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_fedrep.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_pfedme.yaml" &
# narci
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_ditto.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_fedavg.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_fedrep.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_pfedme.yaml" &