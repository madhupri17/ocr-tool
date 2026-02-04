import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Enable GUI display
import matplotlib.pyplot as plt
# plt.style.use('./deeplearning.mplstyle')
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
i = 0 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    
print(f"${cost_1200sqft:.0f} thousand dollars")
plt.show()

# Define cost function
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    """
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

# Plot cost vs w
w_range = np.linspace(0, 400, 100)
costs = []
b_fixed = 100

for w_val in w_range:
    cost = compute_cost(x_train, y_train, w_val, b_fixed)
    costs.append(cost)

plt.figure(figsize=(10, 5))
plt.plot(w_range, costs, c='b', label='Cost Function')
plt.scatter([w], [compute_cost(x_train, y_train, w, b_fixed)], marker='x', c='r', s=100, label=f'Current w={w}')
plt.xlabel('Weight (w)')
plt.ylabel('Cost')
plt.title('Cost vs Weight (w)')
plt.legend()
plt.show()
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
plt.close('all') 

# Plot the new training data
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, marker='x', c='r', label='Training data')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.title('Updated Training Data')
plt.legend()
plt.show()
