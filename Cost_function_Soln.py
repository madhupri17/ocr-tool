import numpy as np
import matplotlib.pyplot as plt
# from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
# plt.style.use('./deeplearning.mplstyle')
# deeplearning.mplstyle

x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])       #(price in 1000s of dollars)

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

# Set model parameters
w = 100
b = 100

# Compute predictions
m = x_train.shape[0]
f_wb = np.zeros(m)
for i in range(m):
    f_wb[i] = w * x_train[i] + b

# Plot training data and prediction line
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, marker='x', c='r', label='Training data')
plt.plot(x_train, f_wb, c='b', label=f'Prediction (w={w}, b={b})')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.title('Training Data and Prediction')
plt.legend()
plt.show()

# Plot cost vs w
w_range = np.linspace(0, 400, 100)
costs = []
b_fixed = 100

for w_val in w_range:
    cost = compute_cost(x_train, y_train, w_val, b_fixed)
    costs.append(cost)

plt.figure(figsize=(10, 5))
plt.plot(w_range, costs, c='b')
plt.scatter([w], [compute_cost(x_train, y_train, w, b_fixed)], marker='x', c='r', s=100, label=f'Current w={w}')
plt.xlabel('Weight (w)')
plt.ylabel('Cost')
plt.title('Cost vs Weight (w)')
plt.legend()
plt.show()