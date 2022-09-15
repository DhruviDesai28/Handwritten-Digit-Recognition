'''LINEAR ALGEBRA PROJECT

HANDWRITTEN DIGIT RECOGNITION

GROUP MEMBERS NAME:-
AU1940272 DHRUVI DESAI
AU1940212 GRISHIKA SHARMA
AU1940293 PRIYANKA PATEL
AU1940172 AAYUSHI CHAUHAN
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# HANDWRITTEN DIGIT RECOGNITION USING LOGISTIC REGRESSION

'''
TO FIND A LOCAL MAXIMUM OF ANY FUNCTION WE USE GRADIENT ASCENT. 

THE GRADIENT ASCENT AIMS AT MAXIMIZING THE OBJECTIVE FUNCTION:- θj ← θj + α*(∂/∂θj)*J(θ)

THE FOLLOWING IS THE EQUATION OF GRADIENT ASCENT ALGORITHM:- θ+ = θ- + α(yi - h(xi))*xbar

ALPHA IS AN SCALING PARAMETER THAT IS USUALLY LESS THAN 1

THE MAXIMIZE LOG FUNCTION IS:-

J(x,θ,y) = Σ^m to i=1 yi*(log(h(xi)) + (1 - yi)*log(1 - h(xi))

OUR HYPOTHESIS OF A SIGMOID FUNCTION IS:- h(xi) = 1/(1 + e^(θT*xbar))

THE FOLLOWING CODE LOGIC OF BATCH GRADIENT ASCENT:-
FOR j FROM 0 -> Max iteration: 
    FOR i FROM 0 -> m: 
        theta += (alpha) * (y[i] - h(x[i])) * xbar
    ENDLOOP
ENDLOOP 

'''

#HERE WE ARE LOADING THE DIGITS FROM 0 TO 9
digits = load_digits()

from matplotlib import pyplot as plt
print("")
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(10):
    ax = fig.add_subplot(4,4, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
plt.show()
    
    
class LogisticRegression():
    """Class for training and using a model for logistic regression"""
    
    """
    cost = 1/m*(h*log(x) - (1-h)*log(1-x))    
    
    """   
    #ALPHA IS AN SCALING PARAMETER THAT IS USUALLY LESS THAN 1

    def set_values(self, initial_params, alpha=0.01, max_iter=1000, class_of_interest=0):
      
        """Set the values for initial params, step size, maximum iteration, and class of interest"""
        self.params = initial_params
        self.alpha = alpha
        self.max_iter = max_iter
        self.class_of_interest = class_of_interest
    
    @staticmethod
    #SIGMOID FUNCTION IS:- h(xi) = 1/(1 + e^(θT*xbar))
    def _sigmoid(x):
      
        """SIGMOID FUNCTION"""
        
        return 1.0 / (1.0 + np.exp(-x))
    #h(xi) WILL KNOW BY _sigmoid(x)
    
    def predict(self, x_bar, params):
      
        """HERE WE ARE PREDICTING THE PROBABILITY OF A CLASS"""  
                
        return self._sigmoid(np.dot(params, x_bar))
    
    def _compute_cost(self, input_var, output_var, params):
      
        """HERE WE ARE COMPUTING THE LOG LIKELIHOOD COST"""
        
        cost = 0
        for x, y in zip(input_var, output_var):
            x_bar = np.array(np.insert(x, 0, 1))
            y_hat = self.predict(x_bar, params)
            
            y_binary = 1.0 if y == self.class_of_interest else 0.0
            cost += y_binary * np.log(y_hat) + (1.0 - y_binary) * np.log(1 - y_hat)
            
        return cost
    
    def train(self, input_var, label, print_iter = 1000):
      
        """HERE WE ARE TRAINING THE MODEL USING BATCH GRADIENT ASCENT"""
        
        iteration = 1
        while iteration < self.max_iter:
            if iteration % print_iter == 0:
                print(f'The Iteration is: {iteration}')
                print(f'The Cost is: {self._compute_cost(input_var, label, self.params)}')
                print('--------------------------------------------')
            
            for i, xy in enumerate(zip(input_var, label)):
                x_bar = np.array(np.insert(xy[0], 0, 1))
                y_hat = self.predict(x_bar, self.params)
                
                y_binary = 1.0 if xy[1] == self.class_of_interest else 0.0
                gradient = (y_binary - y_hat) * x_bar
                self.params += self.alpha * gradient
            
            iteration +=1
        
        return self.params

    def test(self, input_test, label_test):
      
        """HERE WE ARE TESTING THE ACCURACY OF THE MODEL USING TEST DATA"""
        self.total_classifications = 0
        self.correct_classifications = 0
        
        for x,y in zip(input_test, label_test):
            self.total_classifications += 1
            x_bar = np.array(np.insert(x, 0, 1))
            y_hat = self.predict(x_bar, self.params)
            y_binary = 1.0 if y == self.class_of_interest else 0.0
            
            if y_hat >= 0.5 and  y_binary == 1:
                # correct classification of class_of_interest
                self.correct_classifications += 1
              
            if y_hat < 0.5 and  y_binary != 1:
                # correct classification of an other class
                self.correct_classifications += 1
                
        self.accuracy = self.correct_classifications / self.total_classifications
            
        return self.accuracy


digits_train, digits_test, digits_label_train, digits_label_test =train_test_split(digits.data, digits.target, test_size=0.20)


alpha = 1e-2
params_0 = np.zeros(len(digits.data[0]) + 1)

max_iter = 1000
digits_regression_model_0 = LogisticRegression()
digits_regression_model_0.set_values(params_0, alpha, max_iter, 0)

params =digits_regression_model_0.train(digits_train / 16.0, digits_label_train, 500)


digits_accuracy = digits_regression_model_0.test(digits_test / 16.0, digits_label_test)
print(f'Accuracy of prediciting a ZERO digit in test set: {digits_accuracy}')

# LINEAR REGRESSION
'''
TO FIND A LOCAL MINIMUM OF ANY FUNCTION WE USE GRADIENT DESCENT. 

GRADIENT DESCENT AIMS AT MINIMIZING THE OBJECTIVE FUNCTION:- θj ← θj − α*(∂/∂θj)*J(θ)

THE FOLLOWING IS THE EQUATION OF GRADIENT DESCENT ALGORITHM:- θ+ = θ- + (α/m)*(yi - h(xi))*xbar

ALPHA IS AN SCALING PARAMETER THAT IS USUALLY LESS THAN 1

THE MINIMIZE LOG FUNCTION IS:-

J(x,θ,y) = (1/2m)*(Σ^m to i=1 (h(xi)-yi)^2)
HERE h(xi) = θT*xbar

THE FOLLOWING CODE LOGIC OF BATCH GRADIENT DESCENT:-
FOR j FROM 0 -> Max iteration: 
    FOR i FROM 0 -> m: 
        theta += (alpha/m) * (y[i] - h(x[i])) * xbar
    ENDLOOP
ENDLOOP 

THE FOLLOWING CODE LOGIC OF STOCHASTIC GRADIENT DESCENT:-

SHUFFLE(x,y)
FOR i FROM 0 -> m:
    theta += (alpha / m) * (y[i] - h(x[i])) * xbar  
ENDLOOP
'''

true_slope = 10.889
true_intercept = 3.456
input_var = np.arange(0.0,100.0)
output_var = true_slope * input_var + true_intercept + 500.0 * np.random.rand(len(input_var))


plt.figure()
plt.scatter(input_var, output_var)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

def compute_cost(input_var, output_var, params):
  
    "COMPUTING LINEAR REGRESSION COST"
    num_samples = len(input_var)
    cost_sum = 0.0
    for x,y in zip(input_var, output_var):
        y_hat = np.dot(params, np.array([1.0, x]))
        cost_sum += (y_hat - y) ** 2
    
    cost = cost_sum / (num_samples * 2.0)
    
    return cost

def lin_reg_batch_gradient_descent(input_var, output_var, params, alpha, max_iter):
  
    """COMPUTING THE PARAMS FOR LINEAR REGRESSION USING BATCH GRADIENT DESCENT""" 
    iteration = 0
    num_samples = len(input_var)
    cost = np.zeros(max_iter)
    params_store = np.zeros([2, max_iter])
    
    while iteration < max_iter:
        cost[iteration] = compute_cost(input_var, output_var, params)
        params_store[:, iteration] = params
        
        print('--------------------------')
        print(f'The Iteration is: {iteration}')
        print(f'The Cost is: {cost[iteration]}')
        
        for x,y in zip(input_var, output_var):
            y_hat = np.dot(params, np.array([1.0, x]))
            gradient = np.array([1.0, x]) * (y - y_hat)
            params += alpha * gradient/num_samples
            
        iteration += 1
    
    return params, cost, params_store

"""HERE WE ARE TRAINING THE MODEL"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_var, output_var, test_size=0.20)

params_0 = np.array([20.0, 80.0])

alpha_batch = 1e-3
max_iter = 500
params_hat_batch, cost_batch, params_store_batch =\
    lin_reg_batch_gradient_descent(x_train, y_train, params_0, alpha_batch, max_iter)

def lin_reg_stoch_gradient_descent(input_var, output_var, params, alpha):
  
    """COMPUTING THE PARAMS FOR LINEAR REGRESSION USING STOCHASTIC GRADIENT DESCENT"""
    
    num_samples = len(input_var)
    cost = np.zeros(num_samples)
    params_store = np.zeros([2, num_samples])
    
    i = 0
    for x,y in zip(input_var, output_var):
        cost[i] = compute_cost(input_var, output_var, params)
        params_store[:, i] = params
        
        print('--------------------------')
        print(f'The Iteration is: {i}')
        print(f'The Cost is: {cost[i]}')
        
        y_hat = np.dot(params, np.array([1.0, x]))
        gradient = np.array([1.0, x]) * (y - y_hat)
        params += alpha * gradient/num_samples
        
        i += 1
            
    return params, cost, params_store

alpha = 1e-3
params_0 = np.array([20.0, 80.0])
params_hat, cost, params_store =\
lin_reg_stoch_gradient_descent(x_train, y_train, params_0, alpha)

plt.figure()
plt.scatter(x_test, y_test)
plt.plot(x_test, params_hat_batch[0] + params_hat_batch[1]*x_test, 'g', label='BATCH')
plt.plot(x_test, params_hat[0] + params_hat[1]*x_test, '-r', label='STOCHASTIC')
plt.xlabel('xaxis')
plt.ylabel('yaxis')
plt.legend()
plt.show()
print(f'BATCH      T0, T1: {params_hat_batch[0]}, {params_hat_batch[1]}')
print(f'STOCHASTIC T0, T1: {params_hat[0]}, {params_hat[1]}')

#THE FOLLOWING IS THE METHOD FOR RMS(ROOT MEAN SQUARE) USED FOR BOTH BATCH AND STOCHASTIC
rms_batch = np.sqrt(np.mean(np.square(params_hat_batch[0] + params_hat_batch[1]*x_test - y_test)))
rms_stochastic = np.sqrt(np.mean(np.square(params_hat[0] + params_hat[1]*x_test - y_test)))
print(f'BATCH rms:      {rms_batch}')
print(f'STOCHASTIC rms: {rms_stochastic}')

plt.figure()
plt.plot(np.arange(max_iter), cost_batch, 'r', label='BATCH')
plt.plot(np.arange(len(cost)), cost, 'g', label='STOCHASTIC')
plt.xlabel('ITERATION')
plt.ylabel('NORMALIZED COST')
plt.legend()
plt.show()
print(f'Our minimum cost with BGD is: {np.min(cost_batch)}')
print(f'Our minimum cost with SGD is: {np.min(cost)}')