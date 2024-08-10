import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

df = pd.read_csv('customer_purchase_data.csv')
x = df[['Age','AnnualIncome','NumberOfPurchases','ProductCategory','TimeSpentOnWebsite','DiscountsAvailed']].values
y = df['PurchaseStatus'].values.reshape(-1,1)

#z-core normalisation
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_b = np.c_[np.ones(x.shape[0]),x]
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

def init_params(n_features):
    params = np.zeros((n_features+1,1))
    return params

def back_prop(x_b, a, y):
    m = x_b.shape[0]
    e = a - y
    dw1 = (1/m) * np.dot(e.T, x_b[:, 1].reshape(-1, 1))
    dw2 = (1/m) * np.dot(e.T, x_b[:, 2].reshape(-1, 1))
    dw3 = (1/m) * np.dot(e.T, x_b[:, 3].reshape(-1, 1))
    dw4 = (1/m) * np.dot(e.T, x_b[:, 4].reshape(-1, 1))
    dw5 = (1/m) * np.dot(e.T, x_b[:, 5].reshape(-1, 1))
    dw6 = (1/m) * np.dot(e.T, x_b[:, 6].reshape(-1, 1))
    db = (1/m) * e.sum()
    return db, dw1, dw2, dw3, dw4, dw5, dw6

def update_params(db,dw1,dw2,dw3,dw4,dw5,dw6,params,learning_rate=0.01):
    params[0] -= learning_rate*db
    params[1] -= learning_rate*dw1.reshape(1)
    params[2] -= learning_rate*dw2.reshape(1)
    params[3] -= learning_rate*dw3.reshape(1)
    params[4] -= learning_rate*dw4.reshape(1)
    params[5] -= learning_rate*dw5.reshape(1)
    params[6] -= learning_rate*dw6.reshape(1)

    return params

def forward_prop(x_b,params,y):
    m = x.shape[0]
    z = x_b.dot(params)
    a = sigmoid(z)
    cost = (-1/m)*np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    return a,cost

def logistic_regression(x_b,y,n_iterations=1500):
    params = init_params(6)
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        a,cost = forward_prop(x_b,params,y)
        db,dw1,dw2,dw3,dw4,dw5,dw6 = back_prop(x_b,a,y)
        params = update_params(db,dw1,dw2,dw3,dw4,dw5,dw6,params,learning_rate=0.03)
        cost_history[i] = cost
        if i%100==0:
            print(f'The cost in {i}th iteration is: {cost}')
    return params,cost_history

def predict(x_b,params):
    z = x_b.dot(params)
    a = sigmoid(z)
    predicted_values = (a>0.5).astype(int)
    return predicted_values
params,cost_history = logistic_regression(x_b,y)
plt.plot(cost_history)
plt.title('Learning curve')
plt.xlabel('#iterations')
plt.ylabel('Cost')
plt.show()