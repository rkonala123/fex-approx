import numpy as np
import csv
from scipy.special import jn,jv
from scipy.spatial import KDTree

order = 0

# Create a 2D grid of points
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)

# Convert Cartesian coordinates to polar coordinates for Bessel function
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Compute the Bessel function values
Z = jv(order, R)

# Plotting
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')

# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Bessel function value')
# ax.set_title('2D Cylindrical Bessel Function (Order 0)')

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn

def enhanced_sampling_with_values(x_range, y_range, num_samples, function, threshold=0.1, origin_density=50):
    """
    Perform enhanced sampling on the function within the given range and return sampled points with their function values.
    """
    # Uniform sampling
    x = np.linspace(*x_range, num_samples)
    y = np.linspace(*y_range, num_samples)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Z = function(R)

    # Find significant changes
    Z_diff = np.abs(np.diff(Z, axis=0)) > threshold
    significant_points = np.argwhere(Z_diff)

    # Collect significant points and their function values
    sampled_points = {}
    for point in significant_points:
        x_val = X[point[0], point[1]]
        y_val = Y[point[0], point[1]]
        sampled_points[(x_val, y_val)] = function(np.sqrt(x_val**2 + y_val**2))

    # Dense sampling near the origin
    x_dense = np.linspace(-1, 1, origin_density)
    y_dense = np.linspace(-1, 1, origin_density)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    for i in range(origin_density):
        for j in range(origin_density):
            x_val = X_dense[i, j]
            y_val = Y_dense[i, j]
            sampled_points[(x_val, y_val)] = function(np.sqrt(x_val**2 + y_val**2))

    return sampled_points

# Bessel function setup
x_range = (-10, 10)
y_range = (-10, 10)
order = 0
bessel_function = lambda r: jn(order, r)

# Perform enhanced sampling and get function values
sampled_points_with_values = enhanced_sampling_with_values(x_range, y_range, 100, bessel_function)

# # Plotting sampled points and their Bessel function values
# plt.figure(figsize=(8, 6))
# plt.scatter(*zip(*sampled_points_with_values.keys()), c=list(sampled_points_with_values.values()), cmap='viridis', s=1)
# plt.colorbar(label='Bessel Function Value')
# plt.title('Sampled Points with Bessel Function Values for 2D Bessel Function')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)
# plt.show()

coordinates_list = []
function_values_list = []

# Iterate over the dictionary to populate the lists
for coordinates, function_value in sampled_points_with_values.items():
    coordinates_list.append(coordinates)  # Append the coordinate tuple
    function_values_list.append(function_value)  # Append the function value

# Convert lists to NumPy arrays
X = np.array(coordinates_list).reshape(-1,2)  # Array of [x1, x2] pairs
y = np.array(function_values_list).reshape(-1,1)

data = np.hstack((X, y))
with open('bessel_2d_{}.csv'.format(order), 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(data)