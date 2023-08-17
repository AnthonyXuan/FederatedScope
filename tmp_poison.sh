FOLDER_PATH="100-scripts-poison-2"

nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_0.05.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_0.1.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_0.3.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/badnet_0.5.yaml" &

nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_0.05.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_0.1.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_0.3.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/hk_0.5.yaml" &

nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_0.05.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_0.1.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_0.3.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/narci_0.5.yaml" &

nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_0.05.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_0.1.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_0.3.yaml" &
nohup python federatedscope/main.py --cfg "$FOLDER_PATH/signal_0.5.yaml" &