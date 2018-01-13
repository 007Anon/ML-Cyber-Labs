import numpy as np
import matplotlib.pyplot as plt

##############Functions
def ComputeCost(X, y, theta, m):
    h= X.dot(theta)
    J= (1/(2*m))* (np.sum((h-y)**2))
    return(J)

def GradientDescent(X, y, theta, alpha, num_iters, m):
    J_history= np.zeros(num_iters).reshape(num_iters,1)
    for iter in range(num_iters):
        h= X.dot(theta)
        theta-= (alpha/m)*(X.T.dot((h-y)))
        J_history[iter] = ComputeCost(X, y, theta, m)
    return(theta)

##################### Plotting 
print("Plotting Data...")
data= np.loadtxt('data.txt')
m= data.shape[0]
print(m)
X = data[:, 0].reshape(m,1); y= data[:, 1].reshape(m,1)

plt.plot(X, y, 'rx')
plt.xlabel('X'); plt.ylabel('y')

################### Cost and Gradient Descent

X= np.append(np.ones((m,1)), X, axis=1)
theta= np.zeros((2,1))
iterations= 1500
alpha= 0.01

print('Testing the cost function......')

J= ComputeCost(X, y, theta, m)
print('With theta = [0 ; 0]\nCost computed = {0}\n'.format(J))
print('Expected cost value (approx) 32.07\n')
J = ComputeCost(X, y, [[-1],[2]], m)
print('\nWith theta = [-1 ; 2]\nCost computed = {0}\n'.format(J))
print('Expected cost value (approx) 54.24\n')

print('Running Gradient Descent....')
theta = GradientDescent(X, y, theta, alpha, iterations, m)

print('Theta found by gradient descent:\n')
print('{0}\n'.format(theta))
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

plt.plot(X[:,1].reshape(m,1), X.dot(theta), '-')

print('For population = 35,000, we predict a profit of {0}\n'.format(
        np.array([1, 3.5]).dot(theta)*10000))
print('For population = 70,000, we predict a profit of {0}\n'.format(
        np.array([1, 7.0]).dot(theta)*10000))
