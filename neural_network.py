from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
def get_fashion_mnist_dataloaders(val_percentage=0.1, batch_size=1, num_workers=4):
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
    x = x.view(-1, 784)
    y = torch.FloatTensor(len(y), 10).zero_().scatter_(1,y.view(-1,1),1)
    return x, y
    

# call this once first to download the datasets
_ = get_fashion_mnist_dataloaders()


class Logger:
    def __init__(self):
        self.losses_train = []
        self.losses_valid = []
        self.accuracies_train = []
        self.accuracies_valid = []

    def log(self, accuracy_train=0, loss_train=0, accuracy_valid=0, loss_valid=0):
        self.losses_train.append(loss_train)
        self.accuracies_train.append(accuracy_train)
        self.losses_valid.append(loss_valid)
        self.accuracies_valid.append(accuracy_valid)

    def plot_loss_and_accuracy(self, train=True, valid=True):

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
        print(f"Epoch {len(self.losses_train):2d}, \
                Train:loss={self.losses_train[-1]:.3f}, accuracy={self.accuracies_train[-1]*100:.1f}%, \
                Valid: loss={self.losses_valid[-1]:.3f}, accuracy={self.losses_valid[-1]*100:.1f}%", flush=True)

''' Les fonctions dans cette cellule peuvent avoir les mêmes déclarations que celles de la partie 2''' 


def accuracy(y, y_pred) :
    
    card_D = y.size(dim=0)
 
    y_ = torch.argmax(y, dim = 1)
    y_pred_ = torch.argmax(y_pred, dim = 1)

    card_C = sum((y_ == y_pred_)*1)
    
    acc = card_C/card_D
 
    return acc, (card_C, card_D)





def accuracy_and_loss_whole_dataset(data_loader, model):
    cardinal = 0
    loss     = 0.
    n_accurate_preds  = 0.

    for x, y in data_loader:
        x, y = reshape_input(x, y)
        y_pred                = model.forward(x)
        xentrp                = cross_entropy(y, y_pred)
        _, (n_acc, n_samples) = accuracy(y, y_pred)

        cardinal = cardinal + n_samples
        loss     = loss + xentrp
        n_accurate_preds  = n_accurate_preds + n_acc

    loss = loss / float(cardinal)
    acc  = n_accurate_preds / float(cardinal)

    return acc, loss

def inputs_tilde(x, axis=-1):
    ones =  torch.ones(x.shape[0], 1)
    x_hat2 = torch.column_stack((x, ones))

    return x_hat2

def softmax(x, axis=-1):
    x_ = x - torch.max(x, dim=1).values.unsqueeze(dim=1) #values #+ 1e-9
    e_x = torch.exp(x_)

    softmax = e_x/torch.sum(e_x, dim=1).unsqueeze(dim=1) 
   

    return softmax#softmax #

def cross_entropy(y, y_pred):
    loss = -torch.sum(y * torch.log(y_pred + 1e-9 ))/y.size(dim=0)

    return loss


def softmax_cross_entropy_backward(y, y_pred):

    values = y_pred - y
    return values

def relu_forward(x):
    values =  torch.max(torch.zeros_like(x),x)
    return values

def relu_backward(d_x, x):
    
    value = torch.max(torch.zeros_like(x),x)
    values = torch.where(value > 0, 1, 0)
    
    return values*d_x

class MLPModel:
    def __init__(self, n_features, n_hidden_features, n_hidden_layers, n_classes):
        self.n_features        = n_features
        self.n_hidden_features = n_hidden_features
        self.n_hidden_layers   = n_hidden_layers
        self.n_classes         = n_classes
        
        
        self.params = []
        self.params.append(torch.normal(mean=0, std=((2/(self.n_hidden_features+1))**(1/2)), size=(self.n_features  + 1, self.n_hidden_features) ))

        for index in range(1, self.n_hidden_layers ):
            self.params.append(torch.normal(mean=(0), std=((2/(self.n_hidden_features+1))**(1/2)), size=(self.n_hidden_features + 1, self.n_hidden_features)))

        self.params.append(torch.normal(mean=(0), std=((2/(self.n_hidden_features+1))**(1/2)), size=(self.n_hidden_features + 1, self.n_classes)))
        print(f"Teta params={[p.shape for p in self.params]}")

        

        self.a = [None] * (n_hidden_layers+1) # liste contenant le resultat des multiplications matricielles
        self.h = [None] * (n_hidden_layers+1) # liste contenant le resultat des fonctions d'activations

  
        self.t = 0
        self.m_t = [0 for _ in range(n_hidden_layers+1)] # pour Adam: moyennes mobiles du gradient
        self.v_t = [0 for _ in range(n_hidden_layers+1)] # pour Adam: moyennes mobiles du carré du gradient

        self.b1 =  0.9
        self.b2 = 0.999

    def get_std(self, n_in):
        return (2/n_in)**(1/2)


    def forward(self, x):
        outputs = None
        input = inputs_tilde(x)

        self.input = input

        self.a = [input@self.params[0]] # (n_samples, n_hidden_features)
        self.h = [relu_forward(self.a[0])] # (n_samples, n_hidden_features)


        for i in range(1, self.n_hidden_layers):
            self.a.append(inputs_tilde(self.h[i-1]) @ self.params[i])
            self.h.append(relu_forward(self.a[i]))

        self.a.append(inputs_tilde(self.h[self.n_hidden_layers-1]) @ self.params[self.n_hidden_layers])
        self.h.append(softmax(self.a[self.n_hidden_layers]))


        outputs = self.h[-1]


        return outputs

    def backward(self, y, y_pred):

        delta_out = [None]*(self.n_hidden_layers+1)

        delta_out[self.n_hidden_layers] = softmax_cross_entropy_backward(y, y_pred)

        for i in reversed(range(self.n_hidden_layers)):
            delta_current = delta_out[i+1] @ self.params[i+1].T
            delta_out[i] = relu_backward(delta_current[:, :-1], self.a[i])
        grads = [None]*(self.n_hidden_layers+1)
        grads[0] = self.input.T @ delta_out[0] /y.shape[1]
        for i in range(1, self.n_hidden_layers + 1):
            grads[i] = inputs_tilde(self.h[i-1]).T @ delta_out[i]/y.shape[1]

        return grads

        

    def sgd_update(self, lr, grads):
        for i in range(len(self.params)):
            self.params[i] = self.params[i] - lr * grads[i]

    
    def adam_update(self, lr, grads):

        for i in range(len(self.params)):
            self.t += 1
            self.m_t[i] = self.b1 * self.m_t[i] + (1 - self.b1) * grads[i]
            self.v_t[i] = self.b2 * self.v_t[i] + (1 - self.b2) * torch.square(grads[i])
            m_t_hat = self.m_t[i] / (1 - self.b1 ** (self.t ))
            v_t_hat = self.v_t[i] / (1 - self.b2 ** (self.t ))
            self.params[i] = self.params[i] - lr * m_t_hat / (torch.sqrt(v_t_hat) + 1e-8)



def train(model, lr=0.1, nb_epochs=10, sgd=True, data_loader_train=None, data_loader_val=None):
    best_model = None
    best_val_accuracy = 0
    logger = Logger()

    for epoch in range(nb_epochs+1):

        if epoch > 0:
            for x, y in data_loader_train:
                x, y = reshape_input(x, y)

                y_pred = model.forward(x)
                grads  = model.backward(y, y_pred)
                if sgd:
                  model.sgd_update(lr, grads)
                else:
                  model.adam_update(lr, grads)
        
        accuracy_train, loss_train = accuracy_and_loss_whole_dataset(data_loader_train, model)
        accuracy_val, loss_val = accuracy_and_loss_whole_dataset(data_loader_val, model)
        
        if accuracy_val > best_val_accuracy:
            best_val_accuracy = accuracy_val
            best_model = model 


        logger.log(accuracy_train, loss_train, accuracy_val, loss_val)
        if epoch % 1 == 0: # prints every 5 epochs, you can change it to % 1 for example to print each epoch
          print(f"Epoch {epoch:2d}, \
                  Train:loss={loss_train.item():.3f}, accuracy={accuracy_train.item()*100:.1f}%, \
                  Valid: loss={loss_val.item():.3f}, accuracy={accuracy_val.item()*100:.1f}%", flush=True)

    return best_model, best_val_accuracy, logger




if __name__ == "__main__":
    if torch.cuda.is_available() :
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    depth_list = torch.logspace(1,5, steps=6, base=2, dtype=int).tolist()   # Define ranges in a list
    print(depth_list)

    width_list = torch.logspace(2,5, steps=4, base=4, dtype=int).tolist()   # Define ranges in a list
    print(width_list)
    lr = 0.0001           # Some value
    batch_size = 30   # Some value

    with torch.no_grad():
        for depth in depth_list:
            for width in width_list:
                print("------------------------------------------------------------------")
                print("Training model with a depth of {0} layers and a width of {1} units".format(depth, width))
                data_loader_train, data_loader_val, data_loader_test = get_fashion_mnist_dataloaders(val_percentage=0.1, batch_size=batch_size, num_workers=0)
                
                MLP_model = MLPModel(n_features=784, n_hidden_features=width, n_hidden_layers=depth, n_classes=10)
                _, val_accuracy, _ = train(MLP_model,lr=lr, nb_epochs=10, sgd=False, data_loader_train=data_loader_train, data_loader_val=data_loader_val)
                print(f"validation accuracy = {val_accuracy*100:.3f}")
