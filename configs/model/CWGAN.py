name = 'CWGAN'
device='cuda'
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

in_channels = 1
out_channels = 2
lambda_recon = 100
display_step = 10
lambda_gp = 10
lambda_r1 = 1
