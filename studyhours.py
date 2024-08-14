import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
X = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1,1)
X_b = np.c_[np.ones(9),X]
params = np.zeros((2,1))

def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

def forward_prop(x,params,y):
    m = x.shape[0]
    z = x.dot(params)
    a = sigmoid(z)
    cost = (-1/m)*np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    return a,cost

def back_prop(x,a,y):
    m = x.shape[0]
    error = a-y
    dw = (1/m)*(error.T.dot(x))
    db = (1/m)*(np.sum(error))
    return dw,db

def update_params(params,db,dw,learning_rate=0.01):
    params[0] -= learning_rate*db
    params[1] -= learning_rate*(dw).reshape(1)
    return params
def logistic_regression(X,X_b,y,params,learning_rate=0.01,n_iterations=1000):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        a,cost = forward_prop(X_b,params,y)
        dw,db = back_prop(X,a,y)
        params = update_params(params,db,dw,learning_rate)
        cost_history[i] = cost
        if i % 100 == 0:
            print(f'The cost in {i}th iteration: {cost_history[i]}')
    return params,cost_history

params, cost_history = logistic_regression(X,X_b,y,params,0.1,7000)
z = X_b.dot(params)
def predictor(z):
    a = sigmoid(z)
    y_pred = (a>0.5).astype(int)
    return y_pred
y_hat = predictor(z)
accuracy = accuracy_score(y,y_hat)
print(accuracy)
print(y_hat)
print(y)
plt.subplot(1,2,1)
plt.plot(X.ravel(),sigmoid(z),color='red')
plt.title('The sigmoid function')
plt.subplot(1,2,2)
plt.plot(cost_history)
plt.title('learning curve')
plt.suptitle('Study hour statistics')
plt.show()
