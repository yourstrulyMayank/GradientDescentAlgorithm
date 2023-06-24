""" Gradient Descent for Linear Regression """

# yhat = wx + b
# loss = (y-yhat)**2 / N

import numpy as np

# Initialize some parameters
x = np.random.randn(10,1)
y = 5*x + np.random.rand()
# Parameters
w = 0
b = 0
# Hyperparameter
learning_rate = 0.01

# Create gradient desccent function
def descend(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    # Loss = (y-(wx+b))**2
    for xi, yi in zip(x,y):
        dldw += 2*(yi-(w*xi+b))*(-xi)
        dldb += 2*(yi-(w*xi+b))*(-1)
    # Make an update to w and b
    w = w - learning_rate * (1/N) * dldw
    b = b - learning_rate * (1/N) * dldb
    return w,b

# Iteratively make updates
epochs = 400
for epoch in range(epochs):
    # Run Gradient Descent
    w,b = descend(x,y,w,b,learning_rate)
    yhat = (w*x + b)
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'{epoch} loss is {loss}, parameters w: {w}, b: {b}')
print(x)
print(y)