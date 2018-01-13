import numpy as np
from pandas import Series
import scipy.optimize as op
import matplotlib.pyplot as plt

######################Functions
def PlotData(X, y, show= True):
    p= y==1
    n= y==0
    pos1= X[:, 0:1][p]
    pos2= X[:, 1:2][p]
    neg1 = X[:, 0:1][n]
    neg2 = X[:, 1:2][n]
    
    pos, = plt.plot(pos1, pos2, 'b+')
    neg, =plt.plot(neg1, neg2, 'yo')
    
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend([pos, neg], ['Admitted', 'Not Admitted'])
    if show:
        plt.show()
    else:
        pass
    #end
    
def CostFunc(theta, X, y):
    m= X.shape[0]
    z= X.dot(theta)
    h= (1/(1+ np.exp(-z)))
    cost= (-1/m)*((1-y).T.dot(np.log(1-h)) + y.T.dot(np.log(h)))
    return cost
    #end

def sigmoid(z):
    return(1/(1+ np.exp(-z)))
    #end
    
def Grad(theta, X, y):
    m= X.shape[0]
    z= X.dot(theta)
    h= sigmoid(z)
    grad= (1/m)*X.T.dot(h-y)
    return grad
    #end

def PlotDecisionBoundary(theta, X, y):
    PlotData(X[:,1:], y, show=False)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 2]),  max(X[:, 2])])

        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = [
                np.array([mapFeature2(u[i], v[j]).dot(theta) for i in range(len(u))])
                for j in range(len(v))
            ]
        plt.contour(u,v,z, levels=[0.0])

    # Legend, specific for the exercise
    # axis([30, 100, 30, 100])
    #end

def mapFeature2(X1, X2, degree=6):
    quads = Series([X1**(i-j) * X2**j for i in range(1,degree+1) for j in range(i+1)])
    return Series([1]).append([Series(X1), Series(X2), quads])
    #end

def predict(theta, X):
    p= sigmoid(X.dot(theta))
    return(p>=0.5)
####################Loading and Visualising Data
print('Loading Data.....\n')
data= np.loadtxt("data.txt")
m= data.shape[0]
n= data.shape[1]
X= data[:, 0:2]
y= data[:, 2].reshape(m, 1)

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
PlotData(X, y)

####################Compute Cost and Gradient
X= np.append(np.ones((m,1)), X, axis=1)
theta= np.zeros(n).reshape(n,1)
cost, grad= CostFunc(theta, X, y), Grad(theta, X, y)

print('Cost at initial theta (zeros): \n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

test= [[-24], [0.2], [0.2]]
cost, grad= CostFunc(test, X, y), Grad(test, X, y)
print('\nCost at test theta: \n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

#############Optimizing using fmin
theta= op.fmin(func=CostFunc, x0=np.zeros(3), args=(X, y))
theta= theta.reshape(n,1)

cost= CostFunc(theta, X, y)
print('Cost at theta found by fmin: ', cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n', theta);
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

PlotDecisionBoundary(theta, X, y)

################Prediction and Accuracy
prob= sigmoid(np.array([1, 45, 85]).dot(theta))
print(['For a student with scores 45 and 85, we predict an admission probability of '], prob)
print('Expected value: 0.775 +/- 0.002\n\n')

p = predict(theta, X);

print('Train Accuracy: %f\n', np.mean((p == y)) * 100);
print('Expected accuracy (approx): 89.0\n');
print('\n')