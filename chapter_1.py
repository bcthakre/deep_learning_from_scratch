from numpy import ndarray
import matplotlib.pyplot as plt


def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input array
    '''
    import numpy as np

    return np.power(x,2)


def leaky_relu(x: ndarray) -> ndarray:
    '''
    Apply 'leaky relu' function to each element in the ndarray
    '''

    import numpy as np

    return np.maximum(0.2 * x, x)


def linear_function(x: ndarray) -> ndarray:
    '''
    Example of linear function
    '''

    return 3 * x + 2

def polynomial_function(x : ndarray) -> ndarray:
    '''
    Example of a polynomial function
    
    '''

    return 3 * x ** 4 + 2 * x + 1

def approximate_derivative(f, x: ndarray, h = 1e-5) -> ndarray:
    '''
    Approximate the derivattive of x at a given point
    '''
    return (f(x+h)-f(x)) / h

import numpy as np

x_values = np.array([[1,2,3],[4,5,6]])

polynomial_derivative_values = approximate_derivative(polynomial_function,x_values)

print(polynomial_derivative_values)

# Generating values for plotting
x_range = np.linspace(-2, 3, 400)
polynomial_values = polynomial_function(x_range)
derivative_values = approximate_derivative(polynomial_function, x_range)

# Plotting the function and its derivative
plt.figure(figsize=(10, 6))
plt.plot(x_range, polynomial_values, label='Polynomial Function: 3x^2 + 2x + 1')
plt.plot(x_range, derivative_values, label='Derivative', linestyle='--')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Polynomial Function and Its Derivative')
plt.legend()
plt.grid(True)
plt.show()