import numpy as np
###########Functions
def FeatureNormalize(X, m):
    mu= np.mean(X, axis=0)
    sigma= np.max(X, axis=0)
    X-= mu
    X/= sigma
    return(X)
    
def gradientDescentMulti(X, y, theta, alpha, num_iters, m):
    for i in range(num_iters):
        h= X.dot(theta)
        theta= theta- (alpha/m)*((h-y).T.dot(X)).T
    return(theta)
    
def NormalEqn(X, y, m):
    theta= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return(theta)
    
print('Loading data ...\n')
data=  np.loadtxt('data_multi.txt')
m= data.shape[0]
X= data[:, 0:2]
y= data[:, 2].reshape(m, 1)

print('Normalizing Features ...\n')

X= np.append(np.ones((m, 1)), FeatureNormalize(X, m), axis=1)

##################Gradient Descent
print("Running Gradient Descent....\n")
alpha = 0.01
num_iters = 400
theta = gradientDescentMulti(X, y, np.zeros((3,1)), alpha, num_iters, m)

print('Theta computed from gradient descent: \n')
print(theta)

price= np.array([1, 1650, 3]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n', price)

##############Normal Eqn
theta= NormalEqn(X, y, m)
print("\nTheta Computed from Normal Equation: \n")
print(theta)
price= np.array([1, 1650, 3]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using Normal Equation):\n', price)
