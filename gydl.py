#!/usr/bin/env python
# coding: utf-8

import torch 
from torch.utils import data 
import torchvision
from d2l import torch as d2l 
from IPython import display

class DataGenerator() :

    def __init__(self , batch_size , tag  ) :
        
        self._batch_size = batch_size
        self._tag = tag 
        self._f_table_labels = ['tshirt' , 'trouser' , 'pullover' , 'dress' , 'coat',
            'sandal' , 'shirt' , 'sneaker' , 'bag' , 'ankle boot']
        self._d_table_labels = ['0' , '1' , '2' , '3' , '4',
            '5' , '6' , '7' , '8' , '9']
        self.FASION = 1
        self.DIGIT = 0 
        
        
    #********************************************************************
    #
    # download fasion dataset by MNIST and transfroming it  
    #
    #

    def _loading_fasion_mnist_2(self, resize = None ) :

        trans = [torchvision.transforms.ToTensor()]
        if resize :
            trans.insert( 0 , torchvision.transforms.Resize(resize))
        trans = torchvision.transforms.Compose(trans) 
        
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../data2" , train=True , transform=trans , download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../data2" , train=False , transform=trans , download=True)
        return (data.DataLoader(mnist_train , self._batch_size , shuffle=True , 
            num_workers=self._get_dataloader_workers()) , 
                data.DataLoader(mnist_test , self._batch_size , shuffle = False ,
                    num_workers=self._get_dataloader_workers()))


    #********************************************************************
    #
    # download hand writing digit dataset by MNIST and transfroming it  
    #
    #
    def _loading_digit_data_set(self) :
        l1 = 0.1307
        l2 = 0.3071
        mnist_train =torchvision.datasets.MNIST('./data' , train=True , download=False , 
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((l1,),(l2,)) 
        ]))
        mnist_test = torchvision.datasets.MNIST(root="./data" , train=False , download=False ,             transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()]))
        return (torch.utils.data.DataLoader(mnist_train , self._batch_size , shuffle=True ,                                num_workers=self._get_dataloader_workers()) ,                torch.utils.data.DataLoader(mnist_test , self._batch_size , shuffle=False ,                                num_workers=self._get_dataloader_workers()))
    
    def _get_dataloader_workers(self) : 
        return 4 

    
    def loadingData(self) :
        if self._tag == self.FASION :
            return self._loading_fasion_mnist_2()
        elif self._tag == self.DIGIT :
            return self._loading_digit_data_set()
    
    
    def getLabel(self , labels) :
        if self._tag == self.FASION :
            return [ self.f_table_labels[int(i)] for i in labels]
        if self._tag == self.DIGIT :
            return [ self.d_table_labels[int(i)] for i in labels]

        
    def show_images(self, imgs , num_rows , num_cols , titles=None , scale=1.5) :
        """ show dataset mnist pic"""
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
        d2l.use_svg_display() #use svg pictrue format display 
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
        display.display(self.fig)
        display.clear_output(wait=True)

        
class Train() :
    
    def __init__(self , net, train_iter, test_iter, loss, num_epochs, updater ) :
        self.net = net 
        self.train_iter = train_iter 
        self.test_iter = test_iter 
        self.loss = loss 
        self.num_epochs = num_epochs
        self.updater = updater
        
        self.metric = None
        self.animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
            legend=['train loss', 'train acc', 'test acc'])

    def _train_epoch(self) :
        #just adjust model to tain , not means directly train
        if isinstance(self.net , torch.nn.Module) :
            self.net.train()
        # trains loss degree sum , trains accurary degree sum , sample number  
        self.metric = Accumulator(3)
        for X , y in self.train_iter : 
            y_hat = self.net(X)
            #print(y_hat)
            l = self.loss(y_hat , y)
            #print(l)
            # use inner define 
            if isinstance(self.updater , torch.optim.Optimizer) : 
                self.updater.zero_grad()
                l.mean().backward()
                self.updater.step()
            # use external define 
            else :
                l.sum().backward()
                self.updater(X.shape[0])
            self.metric.add( float(l.sum()) , self._accuracy(y_hat , y) , y.numel())
        return self.metric[0]/self.metric[2] , self.metric[1]/self.metric[2]

    def _accuracy(self, y_hat , y) :
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1 :
            y_hat = y_hat.argmax(axis=1) # return index of argmax with axis = 1 
        cmp = y_hat.type(y.dtype) == y 
        return float(cmp.type(y.dtype).sum())

    def _evaluate_accuracy(self) :
        if isinstance(self.net , torch.nn.Module) :
            self.net.eval() # just evaluate not train
        self.metric = Accumulator(2) 
        with torch.no_grad() :
            for X , y in self.test_iter :
                self.metric.add(self._accuracy(self.net(X) , y) , y.numel())
        return self.metric[0] / self.metric[1]


    def start(self): 

        for epoch in range(self.num_epochs):
            print("train epoch : "+str(epoch))
            train_metrics = self._train_epoch()
            test_acc = self._evaluate_accuracy()
            self.animator.add(epoch + 1 , train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

        

