from torch.optim.lr_scheduler import _LRScheduler

class LearningRateScheduler(_LRScheduler):
    """
    Provides inteface of learning rate scheduler.
    """
    def __init__(self, optimizer, lr):
        pass