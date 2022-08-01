# fonctions pour charger les ensembles de donnees
from cmath import nan
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def get_fashion_mnist_dataloaders(val_percentage=0.1, batch_size=1, num_workers=4):
    '''
    get dataset
    '''
    dataset = FashionMNIST("./dataset", train=True,  download=True, transform=transforms.Compose([transforms.ToTensor()]))
    dataset_test = FashionMNIST("./dataset", train=False,  download=True, transform=transforms.Compose([transforms.ToTensor()]))
    len_train = int(len(dataset) * (1.-val_percentage))
    len_val = len(dataset) - len_train
    dataset_train, dataset_val = random_split(dataset, [len_train, len_val])
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size,shuffle=True,num_workers=num_workers)
    data_loader_val   = DataLoader(dataset_val, batch_size=batch_size,shuffle=True,num_workers=num_workers)
    data_loader_test  = DataLoader(dataset_test, batch_size=batch_size,shuffle=True,num_workers=num_workers)
    return data_loader_train, data_loader_val, data_loader_test

def reshape_input(x, y):
    '''
    reshape input
    '''
    x = x.view(-1, 784)
    y = torch.FloatTensor(len(y), 10).zero_().scatter_(1, y.view(-1,1), 1)
    return x, y
    

# call this once first to download the datasets



# simple logger to track progress during training
class Logger:
    '''
    logger class
    '''
    def __init__(self):
        self.losses_train = []
        self.losses_valid = []
        self.accuracies_train = []
        self.accuracies_valid = []

    def log(self, accuracy_train=0, loss_train=0, accuracy_valid=0, loss_valid=0):
        '''
        Log loss and accuracy
        '''
        self.losses_train.append(loss_train)
        self.accuracies_train.append(accuracy_train)
        self.losses_valid.append(loss_valid)
        self.accuracies_valid.append(accuracy_valid)

    def plot_loss_and_accuracy(self, train=True, valid=True):
        '''
        Plot loss and accuracy
        '''

        assert train and valid, "Cannot plot accuracy because neither train nor valid."

        figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                            figsize=(12, 6))
        
        if train:
            ax1.plot(self.losses_train, label="Training")
            ax2.plot(self.accuracies_train, label="Training")
        if valid:
            ax1.plot(self.losses_valid, label="Validation")
            ax1.set_title("CrossEntropy Loss")
            ax2.plot(self.accuracies_valid, label="Validation")
            ax2.set_title("Accuracy")
        
        for ax in figure.axes:
            ax.set_xlabel("Epoch")
            ax.legend(loc='best')
            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.grid(True, which="major", linestyle='-')
            ax.grid(True, which="minor", linestyle='--', color='lightgrey', alpha=.4)            
  
    def print_last(self):
        '''
        Print last loss and accuracy
        '''
        print(f"Epoch {len(self.losses_train):2d}, \
                Train:loss={self.losses_train[-1]:.3f}, accuracy={self.accuracies_train[-1]*100:.1f}%, \
                Valid: loss={self.losses_valid[-1]:.3f}, accuracy={self.losses_valid[-1]*100:.1f}%", flush=True)


def plot_samples():
    '''
    Plot some samples
    '''
    a, _, _ = get_fashion_mnist_dataloaders()
    num_row = 2
    num_col = 5# plot images
    num_images = num_row * num_col
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i, (x,y) in enumerate(a):
        if i >= num_images:
            break
        ax = axes[i//num_col, i%num_col]
        x = (x.numpy().squeeze() * 255).astype(int)
        y = y.numpy()[0]
        ax.imshow(x, cmap='gray')
        ax.set_title(f"Label: {y}")
        
    plt.tight_layout()
    plt.show()



def accuracy(y, y_pred):
    '''
    Calculate accuracy
    '''
    #  y shape (10, 1)
    # todo : nombre d'éléments à classifier.
    #y = y.numpy()
    #y_pred = y_pred.numpy()
    card_D = y.size(dim=0)
 
    # todo : calcul du nombre d'éléments bien classifiés.
    #y_pred = torch.transpose(y_pred, 0, 1)
    y_ = torch.argmax(y, dim = 1)
    y_pred_ = torch.argmax(y_pred, dim = 1)

    card_C = sum((y_ == y_pred_)*1)
    
    # todo : calcul de la précision de classification.
    acc = card_C/card_D
 
    return acc, (card_C, card_D)

def accuracy_and_loss_whole_dataset(data_loader, model):
    '''
    Calculate accuracy and loss on the whole dataset
    '''
    cardinal = 0
    loss     = 0.
    n_accurate_preds  = 0.

    for x, y in data_loader:
        x, y = reshape_input(x, y)

        #x = x.numpy()    # x shape (batch, 784)
        #y = y.numpy().T  # y shape (10, 1)

        y_pred = model.forward(x)

        xentrp = cross_entropy(y, y_pred)
        _, (n_acc, n_samples) = accuracy(y, y_pred)

        cardinal = cardinal + n_samples
        loss = loss + xentrp
        n_accurate_preds = n_accurate_preds + n_acc

    loss = loss / float(cardinal)
    acc  = n_accurate_preds / float(cardinal)

    return acc, loss

def cross_entropy(y, y_pred): 
    '''
    Calculate cross entropy loss
    '''
    # todo : calcul de la valeur d'entropie croisée.
    
    #loss = -torch.mean(y * torch.log(y_pred +1e-9 )) # y(8, 10) y_pred(10, 8 )
    loss = -torch.sum(y * torch.log(y_pred + 1e-9 ))/y.size(dim=0)

    #print(loss)
    return loss

def softmax(x, axis=-1):
    '''
    Calculate softmax of a tensor along the given axis.
    '''

    x_ = x - torch.max(x, dim=1).values.unsqueeze(dim=1) #values #+ 1e-9
    e_x = torch.exp(x_)

    softmax = e_x/torch.sum(e_x, dim=1).unsqueeze(dim=1) 
   

    return softmax#softmax #
  
def inputs_tilde(x, axis=-1):
    '''
    Transforms the model inputs.
    '''
    # augments the inputs `x` with ones along `axis`


    ones =  torch.ones(x.shape[0], 1)
    x_hat2 = torch.column_stack((x, ones))

    return x_hat2



class LinearModel:
    '''
    Linear model class.
    '''
    def __init__(self, num_features, num_classes):
        '''
        Initialization
        '''
        self.params = torch.normal(0, 0.01, (num_features + 1, num_classes))
        self.b = 0
        
        #self.theta = torch.tensor([0]) 
        #self.momentum = torch.tensor([0.9])
        self.m_t = 0 # pour Adam: moyennes mobiles du gradient
        self.v_t = 0 # pour Adam: moyennes mobiles du carré du gradient
        self.b1 =  0.9
        self.b2 = 0.999
        self.t = 0
        
    def forward(self, x):
        '''
        Forward pass
        '''
    
        inputs = inputs_tilde(x)  # inputs shape (8, 785)

        wx = inputs @ self.params  # (8, 785) x (785, 10)
        outputs = softmax(wx) #  shape (8, 10) 
        return (outputs)

    def get_grads(self, y, y_pred, X): 
        '''
        Calcul des gradients du modèle.
        '''
        x = inputs_tilde(X).t()
        grads =  (x) @ (y_pred - y) / y.size(dim=0)  # (785, 8) * ( (8, 10) - (8, 10)) 
        return grads #  (10, 785)

    def sgd_update(self, lr, grads):
        '''
        Stohastic Gradient Descent.
        '''




        self.params = self.params - lr * grads

    
    def adam_update(self, lr, grads):
        '''
        Adam optimizer.
        '''

        self.t += 1
        self.m_t = self.b1 * self.m_t + (1 - self.b1) * grads
        self.v_t = self.b2 * self.v_t + (1 - self.b2) * grads ** 2
        m_t_hat = self.m_t / (1 - self.b1 ** (self.t ))
        v_t_hat = self.v_t / (1 - self.b2 ** (self.t ))
        self.params = self.params - lr * m_t_hat / (torch.sqrt(v_t_hat) + 1e-8)
        

        pass

def train(model, lr=0.1, nb_epochs=10, sgd=True, data_loader_train=None, data_loader_val=None):
    '''
    Training loop 
    '''
    best_model = None
    best_val_accuracy = 0
    
    best_accuracy = 0
    logger = Logger()

    for epoch in range(1, nb_epochs+1):
        # at epoch 0 evaluate random initial model
        #   then for subsequent epochs, do optimize before evaluation.
        if epoch > 0:
            for x, y in data_loader_train:
                x, y = reshape_input(x, y)  # x (batch, 784) y (batch, 10)

                y_pred = model.forward(x)
                loss = cross_entropy(y, y_pred)  # y (10, 8)

                if str(loss.item()) == 'nan':
                    raise(KeyError)
                grads = model.get_grads(y, y_pred, x)
                if sgd:
                    model.sgd_update(lr, grads)
                else:
                    model.adam_update(lr, grads)
            
        accuracy_train, loss_train = accuracy_and_loss_whole_dataset(data_loader_train, model)
        accuracy_val, loss_val = accuracy_and_loss_whole_dataset(data_loader_val, model)
        print(accuracy_val, loss_val)
        
        if accuracy_val > best_val_accuracy:
            best_model = model.params  # TODO : record the best model parameters and best validation accuracy
            best_val_accuracy = accuracy_val

        logger.log(accuracy_train, loss_train, accuracy_val, loss_val)
        if epoch % 1 == 0: # prints every 5 epochs, you can change it to % 1 for example to print each epoch
          print(f"Epoch {epoch:2d}, \
                  Train: loss={loss_train.item():.3f}, accuracy={accuracy_train.item()*100:.1f}%, \
                  Valid: loss={loss_val.item():.3f}, accuracy={accuracy_val.item()*100:.1f}%", flush=True)

    return best_model, best_val_accuracy, logger


if __name__ == "__main__":

    use_gpu = True
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    batch_size_list = [8, 20, 200, 1000]   # Define ranges in a list
    lr_list = [0.1, 0.01, 0.001]         # Define ranges in a list

    with torch.no_grad():
        for lr in lr_list:
            for batch_size in batch_size_list:
                print("------------------------------------------------------------------")
                print("Training model with a learning rate of {0} and a batch size of {1}".format(lr, batch_size))
                data_loader_train, data_loader_val, data_loader_test = get_fashion_mnist_dataloaders(val_percentage=0.1, batch_size=batch_size, num_workers=0)
                
                model = LinearModel(num_features=784, num_classes=10)
                _, val_accuracy, _ = train(model,lr=lr, nb_epochs=20, sgd=False, data_loader_train=data_loader_train, data_loader_val=data_loader_val)
                print(f"validation accuracy = {val_accuracy*100:.3f}")

