



import torch 
import torchvision
#import d2l 
from d2l import torch as d2l 
from torch.utils import data 
from IPython import display




class Accumulator :
    
    def __init__(self , n) :
        self.data = [0.0] * n 

    def add(self , *args) :
        self.data = [ a + float(b) for a , b in zip(self.data , args)]
    
    def reset(self) :
        self.data = [0.0] * len(self.data)

    def __getitem__(self , indx) :
        return self.data[indx]



class Animator: 
    
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
        ylim=None, xscale='linear', yscale='linear',
            fmts=('r-', 'b--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        #d2l.use_svg_display() #use svg pictrue format display 
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
            # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self , x , y) :
        if not hasattr( y , "__len__") :
            y = [y]
        n = len(y)
        if not hasattr(x , "__len__") :
            x = [x] * n 
        if not self.X :
            self.X = [ [] for _ in range(n)]
        if not self.Y :
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        



#********************************************************************
#
# download fasion dataset by MNIST and transfroming it  
#
#

def loading_fasion_mnist_2(batch_size , resize = None ) :
    
    trans = [torchvision.transforms.ToTensor()]
    if resize :
        trans.insert( 0 , torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans) 
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data2" , train=True , transform=trans , download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data2" , train=False , transform=trans , download=True)
    return (data.DataLoader(mnist_train , batch_size , shuffle=True , 
        num_workers=get_dataloader_workers()) , 
            data.DataLoader(mnist_test , batch_size , shuffle = False ,
                num_workers=get_dataloader_workers()))



#********************************************************************
#
# download hand writing digit dataset by MNIST and transfroming it  
#
#
def loading_digit_data_set(batch_size,train=True  ) :
    assert isinstance(train,bool) 
    
    data_set =torchvision.datasets.MNIST('./data' , train=train , download=False , 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,),(0.3071,)) 
    ]))
    
    dataloader = torch.utils.data.DataLoader(data_set , 
        batch_size , shuffle=train)
    
    return dataloader 



#************************************************************************
#
#
#
def show_images( imgs , num_rows , num_cols , titles=None , scale=1.5) :
    """ """
    figsize = (num_cols * scale , num_rows * scale ) 
    _ , axes = d2l.plt.subplots(num_rows , num_cols , figsize=figsize)
    axes = axes.flatten()

    for i , (ax , img) in enumerate(zip(axes , imgs)) :
        if torch.is_tensor(img) :
            ax.imshow(img.numpy())
        else :            
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles : 
            ax.set_title(titles[i])
        
    d2l.plt.show() #encapsulate show func
    
    return axes 



def get_dataloader_workers() : 
    return 4 


def softmax(X) :
    X_exp = torch.exp(X)
    partition = X_exp.sum( 1 , keepdim = True )
    return X_exp / partition


lr = 0.1
num_inputs = 784
num_outputs = 10 
W = torch.normal( 0 , 0.01 , size=(num_inputs , num_outputs) , requires_grad = True )
b = torch.zeros( num_outputs , requires_grad = True )
batch_size = 18

__recognize__ = 'digit'

if __recognize__ == 'clothes' : 
    train_iter, test_iter = loading_fasion_mnist_2(batch_size)
    def get_labels(labels) :
        table_labels = ['tshirt' , 'trouser' , 'pullover' , 'dress' , 'coat',
            'sandal' , 'shirt' , 'sneaker' , 'bag' , 'ankle boot']
        return [ table_labels[int(i)] for i in labels]

if __recognize__ == 'digit' :
    train_iter = loading_digit_data_set(batch_size )
    test_iter = loading_digit_data_set(batch_size , train=False)
    # digit version     
    def get_labels(labels) :
        table_labels = ['0' , '1' , '2' , '3' , '4',
            '5' , '6' , '7' , '8' , '9']
        return [ table_labels[int(i)] for i in labels]






def net(X) :   
    return softmax( torch.matmul(X.reshape((-1,W.shape[0])) , W) + b )
   

def cross_entropy(y_hat , y) :
    return - torch.log(y_hat[range(len(y_hat)) , y ])


def accuracy( y_hat , y) :
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1 :
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y 
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net , data_iter) :
    if isinstance(net , torch.nn.Module) :
        net.eval() # just evaluate not train
    
    metric = Accumulator(2) 
    with torch.no_grad() :
        for X , y in data_iter :
            metric.add(accuracy(net(X) , y) , y.numel())
    return metric[0] / metric[1]

def sgd(params , batch_size) :
    with torch.no_grad() :
        for param in params :
            #print(param.grad) 
            param -= lr * param.grad/batch_size
            param.grad.zero_()

def updater(batch_size) :
    return sgd([W,b], batch_size)



def train_epoch(net , train_iter , loss , updater) :
    if isinstance(net , torch.nn.Module) :
        net.train()
    metric = Accumulator(3)
    for X , y in train_iter : 
        y_hat = net(X)
        #print(y_hat)
        l = loss(y_hat , y)
        #print(l)
        if isinstance(updater , torch.optim.Optimizer) : 
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else :
            l.sum().backward()
            updater(X.shape[0])
        metric.add( float(l.sum()) , accuracy(y_hat , y) , y.numel())
    return metric[0]/metric[2] , metric[1]/metric[2]



def train(net, train_iter, test_iter, loss, num_epochs, updater): 
    
    #animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    #    legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print(epoch)
        train_metrics = train_epoch(net , train_iter , loss, updater)
        test_acc = evaluate_accuracy(net , test_iter)
    #    animator.add(epoch + 1 , train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc




def dominant_control_3() :
    
    #net = torch.nn.Sequential(torch.nn.Flatten() , torch.nn.Linear(784,10))
    num_epochs = 10
    train(net , train_iter , test_iter , cross_entropy , num_epochs , updater)
    
    for X , y in test_iter :
        break
    #trues = d2l.get_fashion_mnist_labels(y)
    trues = get_labels(y)
    #preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    preds = get_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true , pred in zip(trues , preds )]
    n = 6
    d2l.show_images(X[0:n].reshape((n,28,28)) , 1 , n , titles=titles[0:n])
    print(titles)



def pic_debug() :
    #X , y = next(iter(train_iter))
    
    #show_images(X.reshape(18,28,28) , 2 , 9 , titles=get_labels(y))

    num_epochs=10

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
        legend=['train loss', 'train acc', 'test acc','ss'])
    
    for epoch in range(num_epochs) :
        train_metrics = epoch + 1
        test_acc= epoch + 2
        animator.add(epoch + 1 , (train_metrics,) + (test_acc,))
    
    d2l.plt.show()




if __name__ == "__main__" :
    print('************************')
    #dominant_control_3()
    pic_debug()
