name = 'UNet_32_v1'
device='cpu'
learning_rate = 1e-3
early_stop_patience = 4
max_epochs = 12
tboard = True
dataloader_params ={
        'batch_size' : 32,#'persistent_workers' : True,
        'num_workers': 0,
        'train_ratio': 0.75,
        'valid_ratio': 0.15,
        'shuffle_train': True,
        'shuffle_valid': False,
        'shuffle_test': False
        }
pre_trained_model = None # './model_1.pt'