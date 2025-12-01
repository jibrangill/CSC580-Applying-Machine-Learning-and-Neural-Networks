import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#defining fixed seeds for np and tf
np.random.seed(101)
tf.random.set_seed(101)

#generating 50 random numbers from 0 to 50
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)
n = len(x)

#training data before training
#plt.scatter(x, y)
#plt.title('Training Data')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()

w = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)

learning_rate = 0.01
training_epochs = 1000
optimizer = tf.optimizers.SGD(learning_rate)

for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        y_pred = w * x + b
        cost = tf.reduce_mean(tf.pow(y_pred - y, 2)) / (2 * n)

    gradients = tape.gradient(cost, [w, b])
    optimizer.apply_gradients(zip(gradients, [w, b]))

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}: cost={cost.numpy():.4f}, W={w.numpy():.4f}, b={b.numpy():.4f}')

print('\nFinal Results:')
print(f'Final Cost: {cost.numpy():.4f}')
print(f'Final Weight: {w.numpy():.4f}')
print(f'Final Bias: {b.numpy():.4f}')

#plotting training and fitted data graphs
plt.figure(figsize=(14, 6))

#Subplot for training data
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Training Data', color='blue')
plt.title('Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for training data and fitted line
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Training Data', color='blue')
pred_line = w.numpy() * x + b.numpy()
plt.plot(
    x,
    pred_line,
    color='red',
    linewidth=2,
    label=f'Fitted Line\nLR={learning_rate}, Epochs={training_epochs}, Optimizer=SGD'
)
plt.plot([], [], ' ', label=f'Final Cost:   {cost.numpy():.4f}')
plt.plot([], [], ' ', label=f'Final Weight: {w.numpy():.4f}')
plt.plot([], [], ' ', label=f'Final Bias:   {b.numpy():.4f}')
plt.title('Linear Regression Result')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.show()