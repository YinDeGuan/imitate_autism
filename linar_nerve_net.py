#


import random   
import torch 
import torchvision
#import d2l 
from d2l import torch as d2l 
from torch.utils import data 



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
# no frame implements linar net 
#
#

#****************************************************
#
#
def synthetic_data( w , b , num_examples) :
    """ generate y = Xw + b + noise"""
    X = torch.normal(0 , 1 , (num_examples , len(w)))
    y = torch.matmul(X , w ) + b 
    y += torch.normal(0 , 0.01 , y.shape)
    #print(y) # wee see result 
    return X , y.reshape((-1,1))


#****************************************************
#
#
#
def data_iter(batch_size , features , labels) :
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0 , num_examples , batch_size) :
        batch_indices = torch.tensor( 
            indices[i:min(i+batch_size , num_examples)])
        yield features[batch_indices] , labels[batch_indices]



#*********************************************************
#  
#
def linreg( X , w , b ) :
    return torch.matmul(X , w) + b 


#*************************************************************
#
#
def squared_loss(y_hat , y ) :
    return ( y_hat - y.reshape(y_hat.shape))**2/2



#*****************************************************
#
#
def sgd(params , lr , batch_size) :
    with torch.no_grad() :
        for param in params : 
            param -= lr * param.grad/batch_size
            param.grad.zero_()




#*****************************************************
#
#
def dominant_control() :
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features , labels = synthetic_data(true_w , true_b , 1000)
    #d2l.set_figsize()
    #d2l.plt.scatter(features[:,1].detach().numpy() , labels.detach().numpy(),1)
    #d2l.plt.show()

    #batch_size = 10
    #for X , y in data_iter(batch_size , features , labels) :
    #    print( X , '\n' , y )
    #    break 
    
    w = torch.normal(0 , 0.01 , (2,1) , requires_grad=True)
    b = torch.zeros(1 , requires_grad = True )
    #print(w)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10 

    for epoch in range(num_epochs) :
        for X , y in data_iter(batch_size , features, labels) :
            l = loss( net(X,w,b) , y )
            l.sum().backward() # grad for w , b 
            sgd([w,b] , lr , batch_size )
        with torch.no_grad() :
            train_l = loss( net(features , w , b ) , labels )
            print( f'epoch {epoch + 1 } , loss {float(train_l.mean()):f}')
    


def test1() :
    a = torch.tensor([1.0,2.0])
    a.requires_grad = True 
    b = torch.tensor([3.0,4.0])
    b.requires_grad = True 
    F = a * b 
    print(F)
    F.backward(torch.tensor([1.,1.]))
    print(a.grad)
    a.grad.zero_()
    print(a.grad)







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#
#  frame implements linear net 
#
#


def load_array(data_arrays , batch_size , is_train=True) :
    """data iterator """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset , batch_size , shuffle=is_train)



def dominant_control_2() :
    true_w = torch.tensor([2,-3.4])
    true_b = 4.2
    features , labels = synthetic_data(true_w , true_b , 1000)
    batch_size = 10
    data_iter = load_array((features , labels ) , batch_size)
    #print(next(iter(data_iter)))
    
    net = torch.nn.Sequential(torch.nn.Linear(2,1))
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)
    loss = torch.nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters() , lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs) :
        for X , y in data_iter :
            l = loss(net(X),y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss()



if __name__ == "__main__" :
    print("apes")
    #dominant_control()
    #test1()
    #dominant_control_3()
    
