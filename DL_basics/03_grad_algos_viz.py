import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

def f1(x, y):
    return np.sin(x) ** 2 + np.cos(y) ** 2

def f1_grad(x, y):
    return np.array([2 * np.cos(x) * np.sin(x), -2 * np.sin(y) * np.cos(y)])

def f2(x, y):
    return np.sin(x**2) + np.cos(y**2)

def f2_grad(x, y):
    return np.array([2 * x * np.cos(x**2), -2 * y * np.sin(y**2)])


def plot_3d_function(func, x_range = None, y_range = None,width=800, height=800):
    if x_range is None:
        x_range = np.linspace(-5, 5, 100)
    if y_range is None:
        y_range = np.linspace(-5, 5, 100)
    
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='3D Surface Plot', autosize=False,width=width, height=height,margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

# plot_3d_function(f1, np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))



# # ## 1.Vanilla GD


max_iters = 10
learning_rate = 0.01

def gradient_descent(func, grad, x, y, max_iters, learning_rate):
    x_values = []
    y_values = []
    for i in range(max_iters):
        x_values.append(x)
        y_values.append(y)
        grad_x, grad_y = grad(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
    return x_values, y_values


def plot_gradient_descent(func, x_values, y_values, levels = 25):
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)

    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    plt.figure(figsize=(10, 8))
    contour_filled = plt.contourf(X, Y, Z, levels=levels, cmap='Blues')
    contour_lines = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)
    plt.colorbar(contour_filled)
    
    plt.plot(x_values, y_values, 'ro-', markersize=2)
    plt.clabel(contour_lines, inline=True, fontsize=8)
    # inital point
    plt.plot(x_values[0], y_values[0], 'g*', markersize=10, label='Start')
    # final point
    plt.plot(x_values[-1], y_values[-1], 'r*', markersize=10, label='End')
    plt.title('Contour Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()
import time
from matplotlib.animation import FuncAnimation

def plot_and_perform_gradient_descent(func, grad, x_init, y_init, max_iters, learning_rate):
    x = x_init
    y = y_init
    x_r = np.linspace(-3, 3, 100)
    y_r = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_r, y_r)
    Z = func(X, Y)
    plt.contourf(X, Y, Z, levels=25, cmap='Blues')
    plt.colorbar()
    plt.contour(X, Y, Z, levels=25, colors='black', linewidths=0.5)
    plt.plot(x, y, 'ro', markersize=5)
    plt.plot(x_init, y_init, '^',color = 'black', markersize=10, label='Start')
    plt.show()
    for i in range(max_iters):
        grad_x, grad_y = grad(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        plt.plot(x, y, 'ro', markersize=5)
        # time.sleep(1)
    plt.plot(x, y, 'r*', markersize=10, label='End')
    plt.title('Contour Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()



def animate_gradient_descent(func, grad, x_init, y_init, max_iters, learning_rate):

    x = x_init
    y = y_init
    x_values = [x]
    y_values = [y]

    fig, ax = plt.subplots()
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)
    contour_filled = ax.contourf(X, Y, Z, levels=25, cmap='Blues')
    contour_lines = ax.contour(X, Y, Z, levels=25, colors='black', linewidths=0.5)
    ax.plot(x, y, 'ro', markersize=5)
    ax.plot(x_init, y_init, '^', color='black', markersize=10, label='Start')
    ax.legend()

    def update(i):
        nonlocal x, y
        grad_x, grad_y = grad(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        x_values.append(x)
        y_values.append(y)
        ax.plot(x_values, y_values, 'ro-', markersize=2)
        ax.plot(x, y, 'ro', markersize=5)
        return ax

    ani = FuncAnimation(fig, update, frames=max_iters, repeat=False)
    for i in range(max_iters):
        ani._stop()
    plt.show()


x_init = np.random.uniform(-2, 2)
y_init = np.random.uniform(-2, 2)
# x_values, y_values= gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.01)
# plot_gradient_descent(f1, x_values, y_values)

# plot_and_perform_gradient_descent(f1, f1_grad, x_init, y_init, 1000, 0.01)

animate_gradient_descent(f1, f1_grad, x_init, y_init, 100, 0.01)

# max_iters = 1000
# lr = 0.01


# def gradient_descent(func, grad, x, y, max_iters, learning_rate):
#     x_values = []
#     y_values = []
#     for i in range(max_iters):
#         x_values.append(x)
#         y_values.append(y)
#         grad_x, grad_y = grad(x, y)
#         x -= learning_rate * grad_x
#         y -= learning_rate * grad_y
#     return x_values, y_values

# x_init = np.random.uniform(-2, 2)
# y_init = np.random.uniform(-2, 2)
# xv, yv, = gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res = {'Vanilla GD' : [xv, yv]}


# colours_list = sns.color_palette("bright", 12) 
# # colours_list



# def plot_gradient_descent(func, results , levels = 25):
#     x_range = np.linspace(-3, 3, 100)
#     y_range = np.linspace(-3, 3, 100)

#     X, Y = np.meshgrid(x_range, y_range)
#     Z = func(X, Y)

#     plt.figure(figsize=(10, 8))
#     contour_filled = plt.contourf(X, Y, Z, levels=levels, cmap='Blues')
#     contour_lines = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5)
#     plt.colorbar(contour_filled)
#     plt.clabel(contour_lines, inline=True, fontsize=8)
    
#     for key, value in results.items():
#         x_values, y_values = value
#         plt.plot(x_values, y_values, 'o-', markersize=2, label=key, color=colours_list[list(results.keys()).index(key)])
#         plt.plot(x_values[-1], y_values[-1], '*', markersize=10, label=f'{key} End', color=colours_list[list(results.keys()).index(key)])
        
#     plt.plot(x_values[0], y_values[0], '^',color = 'black', markersize=10, label='Start')
#     plt.title('Contour Plot')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.legend()
#     plt.show()

# plot_gradient_descent(f1, res)


# # ## 2 Moment Based GD


# def moment_based_gradient_descent(func, grad, x, y, max_iters, learning_rate, beta = 0.85):
#     x_values = []
#     y_values = []
#     ux = 0 
#     uy = 0
#     beta = beta 
#     for i in range(max_iters):
#         x_values.append(x)
#         y_values.append(y)
#         grad_x, grad_y = grad(x, y)
#         ux = beta * ux + grad_x
#         uy = beta * uy + grad_x
#         x -= learning_rate * ux
#         y -= learning_rate * uy
#     return x_values, y_values
# xv, yv, = moment_based_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Moment Based GD'] =[xv, yv]
# plot_gradient_descent(f1, res)


# # ## 3 nesterov accelerated GD


# def nesterov_accelerated_gradient_descent(func, grad, x, y, max_iters, learning_rate, beta = 0.85):
#     x_values = []
#     y_values = []
#     ux = 0 
#     uy = 0
#     beta = beta 
#     for i in range(max_iters):
#         x_values.append(x)
#         y_values.append(y)
#         grad_x, grad_y = grad(x - beta * ux, y - beta * uy)
#         ux = beta * ux + learning_rate * grad_x
#         uy = beta * uy + learning_rate * grad_y
#         x -= ux
#         y -= uy
#     return x_values, y_values
# lr = 0.1
# x_init = np.random.uniform(-2, 2)
# y_init = np.random.uniform(-2, 2)
# res['Vanilla GD'] = gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Moment Based GD'] = moment_based_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Nesterov Accelerated GD'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# plot_gradient_descent(f1, res)


# # ## 4. Adaptive GD (AdaGRAD)


# def adagrad(func, grad, x, y, max_iters, learning_rate, beta = 0.85, epsilon = 1e-8):
#     x_values = []
#     y_values = []
#     ux = 0 
#     uy = 0
#     # beta = beta 
#     for i in range(max_iters):
#         x_values.append(x)
#         y_values.append(y)
#         grad_x, grad_y = grad(x, y)
#         ux += grad_x**2
#         uy += grad_y**2
#         x = x - learning_rate / (ux + epsilon)**0.5 * grad_x
#         y = y - learning_rate / (uy + epsilon)**0.5 * grad_y
#     return x_values, y_values

# lr = 0.1
# x_init = np.random.uniform(-2, 2)
# y_init = np.random.uniform(-2, 2)

# res['Vanilla GD'] = gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Moment Based GD'] = moment_based_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Nesterov Accelerated GD'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Adagrad'] = adagrad(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)

# plot_gradient_descent(f1, res)


# # ## 5. RMSprop - Adagrad 


# def adagrad_RMSprop(func, grad, x, y, max_iters, learning_rate, beta = 0.85, gamma = 0.9, epsilon = 1e-8):
#     x_values = []
#     y_values = []
#     ux = 0 
#     uy = 0
#     # beta = beta 
#     gamma = gamma
#     for i in range(max_iters):
#         x_values.append(x)
#         y_values.append(y)
#         grad_x, grad_y = grad(x, y)
#         ux = gamma * ux + (1-gamma)*grad_x**2
#         uy = gamma * uy + (1-gamma)*grad_y**2
#         x = x - learning_rate / (ux + epsilon)**0.5 * grad_x
#         y = y - learning_rate / (uy + epsilon)**0.5 * grad_y
#     return x_values, y_values

# lr = 0.1
# x_init = np.random.uniform(-2,-1 )
# y_init = np.random.uniform(-1, 1)

# res['Vanilla GD'] = gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Moment Based GD'] = moment_based_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Nesterov Accelerated GD'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Adagrad'] = adagrad(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['RMSprop'] = adagrad_RMSprop(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)

# plot_gradient_descent(f1, res)


# # > Observation: RMSprop oscillating here = sensitive to init learning rate $\eta$


# x_init = np.random.uniform(-0.5,0.5)
# y_init = np.random.uniform(-0.5, 0.5)
# xv , yv = adagrad_RMSprop(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.1)
# rd= {'RmsProp 0.1' : [xv, yv]}
# # xv , yv = adagrad_RMSprop(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.3)
# # rd['RmsProp 0.3'] = [xv, yv]
# # xv , yv = adagrad_RMSprop(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.01)
# # rd['RmsProp 0.01'] = [xv, yv]
# # xv , yv = adagrad_RMSprop(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.05)
# # rd['RmsProp 0.05'] = [xv, yv]
# xv , yv = adagrad_RMSprop(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.001)
# rd['RmsProp 0.001'] = [xv, yv]
# plot_gradient_descent(f1,rd)


# # ## 6. AdaDelta


# def adadelta(func, grad, x, y, max_iters, beta = 0.85, gamma = 0.75, epsilon = 1e-6):
#     x_values = []
#     y_values = []
    
#     vx = 0 
#     vy = 0

#     ux = 0
#     uy = 0

#     # beta = beta 
#     gamma = gamma
#     for i in range(max_iters):
#         x_values.append(x)
#         y_values.append(y)
#         grad_x, grad_y = grad(x, y)
#         vx = gamma * vx + (1-gamma)*grad_x**2
#         delta_x = - (ux + epsilon)**0.5 / (vx + epsilon)**0.5 * grad_x
#         vy = gamma * vy + (1-gamma)*grad_y**2
#         delta_y = - (uy + epsilon)**0.5 / (vy + epsilon)**0.5 * grad_y
#         x += delta_x
#         y += delta_y
        
#         ux = gamma * ux + (1-gamma) * delta_x**2
#         uy = gamma * uy + (1-gamma) * delta_y**2
#     return x_values, y_values

# lr = 0.1
# x_init = np.random.uniform(-2,-1 )
# y_init = np.random.uniform(-1, 1)
# res = {}
# # res['Vanilla GD'] = gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# # res['Moment Based GD'] = moment_based_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# # res['Nesterov Accelerated GD'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['Adagrad'] = adagrad(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['RMSprop'] = adagrad_RMSprop(f1, f1_grad, x_init, y_init, max_iters, learning_rate =lr)
# res['AdaDelta'] = adadelta(f1, f1_grad, x_init, y_init, max_iters)

# plot_gradient_descent(f1, res)


# # ## Different learning Rates


# res2 = {}
# max_iters = 1000
# # res2['alpha = 2'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 10)
# res2['alpha = 1'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 1)
# res2['alpha = 0.1'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.1)
# res2['alpha = 0.01'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.01)
# res2['alpha = 0.001'] = nesterov_accelerated_gradient_descent(f1, f1_grad, x_init, y_init, max_iters, learning_rate = 0.001)

# plot_gradient_descent(f1, res2)





