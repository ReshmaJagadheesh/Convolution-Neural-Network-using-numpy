import numpy as np
import h5py  
import copy 
from random import randint

#load MNIST data 
MNIST_data = h5py.File('MNISTdata.hdf5', 'r') 
x_train = np.float32(MNIST_data['x_train'][:]) 
y_train = np.int32(np.array(MNIST_data['y_train'][:,0])) 
x_test = np.float32(MNIST_data['x_test'][:]) 
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
MNIST_data.close()

#dimension of inputs 
n_x = 28
#number of outputs 
n_y = 10
#number of channels
n_c = 5
#Filter size
f = 3

model = {}

model["W1"] = np.random.randn(f,f,n_c) / np.sqrt(n_x)
model["W2"] = np.random.randn(n_y,((n_x-f+1)*(n_x-f+1)*n_c)) / np.sqrt(n_x)
model["b2"] = np.random.randn(n_y,1)

model_grads = copy.deepcopy(model)

def relu(z):
    return z*(z>0)

def drelu(z):
    return 1*(z>0)

def softmax(z): 
    ZZ = np.exp(z)/np.sum(np.exp(z),axis=0) 
    return ZZ

def conv(a,W):
    return np.sum(np.multiply(a,W))

def forward_propagation(x,model,C):
    W1 = model["W1"]
    W2 = model["W2"]
    b2 = model["b2"]
    n_x = 28
    
    x = np.reshape(x,(n_x,n_x))
    f = W1.shape[0]
    
    n_w = n_h = n_x-f+1
    Z1 = np.zeros((n_h,n_w,C))
    
    for h in range(n_h):
        for w in range(n_w):
            for c in range(C):
                vert_start = h
                vert_end = vert_start + f
                horiz_start = w
                horiz_end = horiz_start + f
                
                a = x[vert_start:vert_end,horiz_start:horiz_end]
                Z1[h,w,c] = conv(a,W1[:,:,c])
                
    A1 = relu(Z1)
    A1 = np.reshape(A1,(-1,1))
    Z2 = np.dot(W2,A1)+b2
    A2 = softmax(Z2)
    
    cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
    return A2,cache
    
def backward_propagation(x,y,A2, model, model_grads,cache):
    n_x = 28
    n_y = 10
    y_t = np.zeros((1,n_y))
    y_t[np.arange(1), y] = 1
    y_t = np.reshape(y_t,(-1,1))
    
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    W2 = model["W2"]
    W1 = model["W1"]
    
    delta1 = y_t-A2
    db2 = np.sum(delta1,axis=1,keepdims=True)
    dW2 = np.dot(delta1,A1.T)
    
    delta2 = np.dot(W2.T,delta1)
    delta2 = np.reshape(delta2,Z1.shape)
    delta3 = np.multiply(delta2,drelu(Z1))
    
    n_h = W1.shape[0]
    n_w = W1.shape[1]
    C = W1.shape[2]
    dW1 = np.zeros(W1.shape)
    
    x = np.reshape(x,(n_x,n_x))
    
    for h in range(n_h):
        for w in range(n_w):
            for c in range(C):
                vert_start = h
                vert_end = vert_start + f
                horiz_start = w
                horiz_end = horiz_start + f
                
                a = x[vert_start:vert_end,horiz_start:horiz_end]
                
                dW1[:,:,c] +=  a * delta3[h,w,c]
    
    
    model_grads["W1"] = dW1
    model_grads["W2"] = dW2
    model_grads["b2"] = db2
    
    return model_grads

learning_rate = 0.01
num_epochs = 3

for epochs in range(num_epochs):
    if (epochs > 5): 
        learning_rate = 0.001 
    if (epochs > 10):
        learning_rate = 0.0001 
    if (epochs > 15): 
        learning_rate = 0.00001
        
    total_correct = 0 
    ite = len(x_train)
    
    for n in range(ite): 
        n_random = randint(0,len(x_train)-1) 
        y = y_train[n_random] 
        x = x_train[n_random][:] 
        A2,cache = forward_propagation(x, model,n_c) 
        prediction = np.argmax(A2) 
        if (prediction == y): 
            total_correct += 1 
        model_grads = backward_propagation(x,y,A2, model, model_grads,cache) 
        model["W1"] = model["W1"] + learning_rate*model_grads["W1"] 
        model["W2"] = model["W2"] + learning_rate*model_grads["W2"]
        model["b2"] = model["b2"] + learning_rate*model_grads["b2"]
    print(total_correct/np.float(ite))
    
    #test accuracy
    total_correct = 0 
for n in range(len(x_test)): 
    y = y_test[n] 
    x = x_test[n][:] 
    x = np.reshape(x,(1,-1))
    A2,cache = forward_propagation(x, model,n_c) 
    prediction = np.argmax(A2) 
    if (prediction == y): 
        total_correct += 1

print(total_correct/np.float(len(x_test)))
