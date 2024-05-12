from data import *
from featurizers import NDFeaturizer, MPNNFeaturizer, RPETFeaturizer, MorganFeaturizer
from models import NeuralDevice, MPNN, RPETransformer, FFN
from copy import deepcopy
import io
from torch_geometric.data import Batch


def train_model(model, loss_func, optimizer, max_n_epoch, overfit_patience, 
                loader, X_train, y_train, X_val, y_val, batch_size):

    log_stream = io.StringIO()
    lr_0 = opt.state_dict()['param_groups'][0]['lr']
    best_state = deepcopy(model.state_dict())
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.5,
    )
    min_val_loss = float('inf')
    overfit_counter = 0

    for i in range(max_n_epoch):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for X, y in loader(X_train, y_train, batch_size=batch_size):
            pred = model(*X)
            loss = loss_func(pred, y)
            loss.backward()
            train_loss += loss.item()
            opt.step()
            opt.zero_grad()

        log_stream.write(f'epoch: {i+1}, train_loss: {train_loss}, ')

        model.eval()
        with torch.no_grad():
            for X, y in loader(X_val, y_val, batch_size=batch_size, shuffle=False):
                pred = model(*X)
                loss = loss_func(pred, y)
                val_loss += loss.item()

        log_stream.write(f'val_loss: {val_loss}\n')

        sch.step(val_loss)
        if val_loss < min_val_loss:
            overfit_counter = 0
            min_val_loss = val_loss
            log_stream.write('saving state... ')
            best_state = deepcopy(model.state_dict())
            log_stream.write('saved\n')
        else:
            overfit_counter += 1
        if overfit_counter > overfit_patience:
            log_stream.write('model overfitted!\n')
            break
    else:
        log_stream.write('max_n_epoch reached\n')
    model.load_state_dict(best_state)
    return model, log_stream



class Loader:

    def __init__(self, target_dtype, device):

        self.target_dtype = target_dtype
        self.device = device


class MPNNLoader(Loader):

    def __init__(self, target_dtype, device='cpu', reconstruct=True):
        super.__init__(target_dtype, device)

        self.reconstruct = reconstruct

    def __call__(self, X, y, batch_size=32, shuffle=True):

        N = len(X)
        assert N == len(y)

        if shuffle:
            self.indices = np.random.permutation(N)
        else:
            self.indices = np.arange(N)

        for i in range(0, N, batch_size):
            curr_indices = indices[i:i+batch_size]
            batch_X = [X[idx].to(self.device) for idx in curr_indices]
            batch_X = Batch.from_data_list(batch_X)
            batch_y = torch.tensor([ys[idx] for idx in curr_indices], device=self.device, dtype=self.target_dtype)
            if self.reconstruct:
                batch_y = (batch_y, batch_X.x)
            yield (batch_X.x, batch_X.edge_index, batch_X.edge_attr, batch_X.batch), batch_y


class NDLoader:

    def __init__(self, target_dtype, device='cpu'):
        super.__init__(target_dtype, device)

    def __call__(self, X, y, batch_size=32, shuffle=True):

        N = len(X)
        assert N == len(y)
        K = len(X[0])
        eye_nfs = tuple(x.size()[1] for x in X[0])

        if shuffle:
            indices = np.random.permutation(N)
        else:
            indices = np.arange(N)

        for i in range(0, N, batch_size):
            eye_inputs = [list() for k in range(K)]
            batch_indices = [list() for k in range(K)]
            targets = []
            curr_indices = indices[i:i+batch_size]
            for idx, j in enumerate(curr_indices):
                targets.append(y[j])
                for k in range(K):
                    eye_inputs[k].append(X[j][k])
                    batch_indices[k].extend([idx,]*X[j][k].shape[0])
            batch_X = tuple(torch.cat(eye_inputs[k], dim=0).to(self.device) for k in range(K))
            batch_indices = torch.tensor(batch_indices, device=self.device, dtype=torch.int32)
            batch_y = torch.tensor(targets, device=self.device, dtype=self.target_dtype)
            yield (batch_X, batch_indices), batch_y


class ReconstructLoss(nn.Module):

    def __init__(self, task, coefs=(1., 1.), ce_weight=None):

        if task == 'classification':
            self.task_loss = nn.CrossEntropyLoss(weight=ce_weight)
        elif task == 'regression':
            self.task_loss = nn.MSELoss()
        self.reconstruct_loss = nn.BCEWithLogitsLoss()
        self.coefs = coefs

    def forward(self, inpt, target):

        task_input, reconstruct_input = inpt
        task_target, reconstruct_target = target
        return (self.task_loss(task_input, task_target) * self.coefs[0]
              + self.reconstruct_loss(reconstruct_input, reconstruct_target) * self.coefs[1])


class FFNLoader(Loader):
    
    def __init__(self, target_dtype, device='cpu'):
        super().__init__(target_dtype, device)

    def __call__(self, X, y, batch_size=32, shuffle=True):
        N = len(x)
        assert N == len(y)
        y = np.array(y)

        if shuffle:
            indices = np.random.permutation(N)
        else:
            indices = np.arange(N)

        for i in range(0, N, batch_size):
            curr_indices = indices[i:i+batch_size]
            batch_X = torch.tensor(X[curr_indices], dtype=torch.float, device=self.device)
            batch_y = torch.tensor(y[curr_indices], dtype=self.target_dtype, device=self.device)
            yield batch_X, batch_y
