import logging

logger = logging.getLogger(__name__)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.epoch = 0
        
        

    def early_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(f'Early stopping at epoch {self.epoch} with val_loss: {val_loss:.4f}')
            return False
        return True

