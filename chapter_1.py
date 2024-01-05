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

from typing import Callable, List
import numpy as np

def deriv(func: Callable[[np.ndarray],np.ndarray],
                             input_: ndarray,
                             delta : float = 0.001) -> ndarray:
    """
    Derive the derivattive of function func at every point on the ndarray
    
    """

    return func(input_ + delta ) - func(input_ - delta) / (2 * delta)


x_values = np.linspace(0,2 * np.pi , 100)

numeircal_derivative = deriv(np.sin,x_values)

actual_derivattive = np.cos(x_values)

plt.figure(figsize=(12,6))
plt.plot(x_values, numeircal_derivative, label='Numerical Derivative of sin(x)')
plt.plot(x_values, actual_derivattive, label='Actual Derivative of sin(x) (cos(x))', linestyle='--')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Comparison of Numerical and Actual Derivatives')
plt.legend()
plt.grid(True)
plt.show()

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

def deriv(func: Callable[[np.ndarray],np.ndarray],
          input_: ndarray,
          delta : float = 0.001) -> np.ndarray:
    """
    Evaluates the derivative of function "func" at every element in the "input" array
    """

    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def second_deriv(func: Callable[[np.ndarray],np.ndarray],
                 input_ : np.ndarray,
                 delta : float = 0.001) -> np.ndarray:
      """
      Evaluates the second derivative
      """

      first_derivative = deriv(func, input_, delta)

      return deriv(lambda x: deriv(func,x,delta), input_, delta)


def polynomial_function(x: np.ndarray) -> np.ndarray:
    return 2 * x ** 3 + 3 * x ** 2 + 5 * x 

x_value = np.linspace(-2,2,100)

first_derivative_value = deriv(polynomial_function,x_value)

second_derivative_value = second_deriv(polynomial_function,x_value)

print(x_value)
print(first_derivative_value)
print(second_derivative_value)

derivative_tuples = list(zip(x_value,first_derivative_value,second_derivative_value))
print(derivative_tuples)

# Plotting the original function, first derivative, and second derivative
plt.figure(figsize=(12, 8))
plt.plot(x_values, x_value, label='Original Polynomial Function')
plt.plot(x_values, first_derivative_value, label='First Derivative', linestyle='--')
plt.plot(x_values, second_derivative_value, label='Second Derivative', linestyle='-.')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Polynomial Function and its Derivatives')
plt.legend()
plt.grid(True)
plt.show()


# one more function

# Define a function that works over the given domain, for example, a cubic function
def cubic_function(x: np.ndarray) -> np.ndarray:
    return x**3

# Generate a range of x values that do not include the domain where the original function is not defined
x_values = np.linspace(0, 4, 100)

# Calculate the original function values, first and second derivatives for the cubic function
original_values = cubic_function(x_values)
first_derivative_values = deriv(cubic_function, x_values)
second_derivative_values = second_deriv(cubic_function, x_values)

# Print statements
print(x_values)
print(first_derivative_values)
print(second_derivative_values)

derivative_tuples = list(zip(x_values, original_values, first_derivative_values, second_derivative_values))
print(derivative_tuples)

# Plotting the original function, first derivative, and second derivative
plt.figure(figsize=(12, 8))
plt.plot(x_values, original_values, label='Original Function')
plt.plot(x_values, first_derivative_values, label='First Derivative', linestyle='--')
plt.plot(x_values, second_derivative_values, label='Second Derivative', linestyle='-.')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Function and its Derivatives')
plt.legend()
plt.grid(True)
plt.show()


# Define a function that works over the given domain, for example, a cubic function
def sqrt_function(x: np.ndarray) -> np.ndarray:
    return x**2

# Generate a range of x values that do not include the domain where the original function is not defined
x_values = np.linspace(0, 100, 1000)

# Calculate the original function values, first and second derivatives for the cubic function
original_values = sqrt_function(x_values)
first_derivative_values = deriv(sqrt_function, x_values)
second_derivative_values = second_deriv(sqrt_function, x_values)

# Print statements
print(x_values)
print(first_derivative_values)
print(second_derivative_values)

derivative_tuples = list(zip(x_values, original_values, first_derivative_values, second_derivative_values))
print(derivative_tuples)

# Plotting the original function, first derivative, and second derivative
plt.figure(figsize=(12, 8))
plt.plot(x_values, original_values, label='Original Function')
plt.plot(x_values, first_derivative_values, label='First Derivative', linestyle='--')
plt.plot(x_values, second_derivative_values, label='Second Derivative', linestyle='-.')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Function and its Derivatives')
plt.legend()
plt.grid(True)
plt.show()


# Chain rule implemtation and graphics


from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np


def plot_chain(chain: list, input_range: np.ndarray) -> None:
    """
    Plot the composite function f2(f1(x)) for a given input range.
    """
    f1, f2 = chain
    composite_function = f2(f1(input_range))
    
    plt.figure(figsize=(10, 6))
    plt.plot(input_range, composite_function, label='Composite Function: $f_2(f_1(x))$')
    plt.xlabel('x')
    plt.ylabel('f2(f1(x))')
    plt.title('Composite Function Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_chain_deriv(chain: list, input_range: np.ndarray) -> None:
    """
    Plot the derivative of the composite function f2(f1(x)) for a given input range.
    """
    derivative = chain_derivative(chain, input_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(input_range, derivative, label='Derivative of Composite Function', color='orange')
    plt.xlabel('x')
    plt.ylabel("Derivative of $f_2(f_1(x))$")
    plt.title('Composite Function Derivative Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid function to each element in the input noarray
    """
    return 1 / (1 + np.exp(-x))

def square(x: np.ndarray) -> np.ndarray:
    """
    Square each element in the input ndarray
    """
    return np.power(x,2)


def chain_derivative(chain: List[Callable[[np.ndarray], np.ndarray]], 
                     input_range: np.ndarray) -> np.ndarray:
    """
    Uses the chain rule to coompute the derivative of two nested functions
    (f2(f1(x))' = f2'(f1(x)) * f1'(x))
    
    """

    assert len(chain) == 2, \
    "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, \
    "function requires a 1 dimensional ndarray as input range"

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx

    f1_of_x = deriv(f1,input_range)

    # df1/du
    df1dx = deriv(f2,f1(input_range))

    # df2/du(f1(x))

    df2du = deriv(f2,f1(input_range))

    # multiplying these quantities together ar each point

    return df1dx * df2du


# Example usage

PLOT_RANGE = np.arange(-3, 3, 0.01)
chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

plot_chain(chain_1, PLOT_RANGE) 
plot_chain_deriv(chain_1, PLOT_RANGE)

plot_chain(chain_2, PLOT_RANGE)
plot_chain_deriv(chain_2, PLOT_RANGE)