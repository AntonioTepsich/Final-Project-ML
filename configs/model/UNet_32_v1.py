name = 'UNet_32_v1'
device='cuda'
learning_rate = 2e-4
early_stop_patience = 4
max_epochs = 30
tboard = True
dataloader_params ={
        'batch_size' : 32,#'persistent_workers' : True,
        'num_workers': 0,
        'train_ratio': 0.8,
        'valid_ratio': 0.1,
        'shuffle_train': True,
        'shuffle_valid': False,
        'shuffle_test': False
        }
pre_trained_model = None # 'checkpoint_10.pt'  #Por default None carga el "full_model.pt"
