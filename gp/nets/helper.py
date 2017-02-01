import numpy as np

from nolearn.lasagne import BatchIterator

class MyTestBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        
        # regularize the batch (which is already in the range 0..1)
        if isinstance(Xb, dict):
            # this is for our multi-leg CNN

            for k in Xb:
                Xb[k] = (Xb[k] - .5).astype(np.float32)

        else:

            Xb = Xb - .5
            
        return Xb, yb

class MyBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(MyBatchIterator, self).transform(Xb, yb)

        # regularize the batch (which is already in the range 0..1)
        if isinstance(Xb, dict):
            # this is for our multi-leg CNN

            for k in Xb:
                Xb[k] = (Xb[k] - .5).astype(np.float32)

        else:

            Xb = Xb - .5
 
        # rotate each patch randomly
        k_s = np.array([0,1,2,3],dtype=np.uint8)
        if isinstance(Xb, dict):
            # this is for our multi-leg CNN

            for i in range(len(Xb['image_input'])):
                k = np.random.choice(k_s)
                for key in Xb:
                    Xb[key][i][0] = np.rot90(Xb[key][i][0], k)

        else:

            for i in range(len(Xb)):
                k = np.random.choice(k_s)
                for j in range(Xb.shape[1]):
                    Xb[j][0] = np.rot90(Xb[j][0], k)

        return Xb, yb

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {0:.6f} at epoch {1}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()
            
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)            
            
def float32(k):
    return np.cast['float32'](k)
