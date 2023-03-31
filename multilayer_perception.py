#

import torch
from d2l import torch as d2l
import gydl



def center_control() :
    x = torch.arange(-8.0 , 8.0 , 0.1 , requires_grad=True)
    y = torch.relu(x)
    d2l.plot( x.detach().numpy() , y.detach() , 'x' , 'relu(x)' , figsize=(5,2.5))
    y.backward(torch.ones_like(x),retain_graph=True)
    d2l.plot(x.detach() , x.grad , 'x' , 'grade of relu' , figsize=(5,2.5))
    #d2l.plt.show()

    
batch_size=256
train_iter , test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs , num_outputs , num_hiddens = 784,10,256
W1 = torch.nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
B1 = torch.nn.Parameter(torch.zeros(num_hiddens, requires_grad=True ))
    # 784 to 256 for what 
W2 = torch.nn.Parameter(torch.randn(num_hiddens , num_outputs , requires_grad=True)*0.01)
B2 = torch.nn.Parameter(torch.zeros(num_outputs, requires_grad=True ))

params = [W1 , B1 , W2 , B2]
num_epochs , lr = 10 , 0.1
loss = torch.nn.CrossEntropyLoss(reduction='none')
updater = torch.optim.SGD(params , lr = lr )

def _relu(X) :
    a = torch.zeros_like(X)
    return torch.max(X,a)

def net(X) :
    X = X.reshape((-1,num_inputs))
    H = _relu(X@W1+B1)
    return (H@W2+B2)

        
def center_control_2() :
    """"""
    gydl.Train(net,train_iter , test_iter,loss, num_epochs,updater).start()
    
    
    

if __name__ == "__main__" :
    print("ss")
    center_control_2() 
