import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Make TF2 behave like TF1
tf.compat.v1.disable_eager_execution()

# Generate synthetic data
N = 100
np.random.seed(0)  # for reproducibility

# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array([-1, -1]), cov=0.1 * np.eye(2), size=(N // 2,)
)
y_zeros = np.zeros((N // 2,))

# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array([1, 1]), cov=0.1 * np.eye(2), size=(N // 2,)
)
y_ones = np.ones((N // 2,))

x_np = np.vstack([x_zeros, x_ones]).astype(np.float32)
y_np = np.concatenate([y_zeros, y_ones]).astype(np.float32)


#Plot x_zeros and x_ones
plt.figure()
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], label='Class 0')
plt.scatter(x_ones[:, 0], x_ones[:, 1], label='Class 1')
plt.title('Synthetic Data (Class 0 vs Class 1)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Generate TensorFlow graph
with tf.name_scope("placeholders"):
    x = tf.compat.v1.placeholder(tf.float32, (N, 2), name="x")
    y = tf.compat.v1.placeholder(tf.float32, (N,), name="y")

with tf.name_scope("weights"):
    W = tf.Variable(tf.random.normal((2, 1)), name="W")
    b = tf.Variable(tf.random.normal((1,)), name="b")

with tf.name_scope("prediction"):
    # Linear logit: shape (N,)
    y_logit = tf.squeeze(tf.matmul(x, W) + b, name="y_logit")

    # Sigmoid gives P(y=1 | x)
    y_one_prob = tf.sigmoid(y_logit, name="y_one_prob")

    # Threshold at 0.5 -> predicted class 0 or 1
    y_pred = tf.round(y_one_prob, name="y_pred")

with tf.name_scope("loss"):
    # Cross entropy for each sample
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_logit, labels=y, name="entropy"
    )
    # Total loss
    l = tf.reduce_sum(entropy, name="loss")

with tf.name_scope("optim"):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(l)

with tf.name_scope("summaries"):
    tf.compat.v1.summary.scalar("loss", l)
    merged = tf.compat.v1.summary.merge_all()

# Summary writer for TensorBoard
train_writer = tf.compat.v1.summary.FileWriter(
    'logistic-train', tf.compat.v1.get_default_graph()
)

#Train the model
num_epochs = 2000  # enough steps to converge on this simple problem

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(num_epochs):
        feed_dict = {x: x_np, y: y_np}
        _, loss_val, summary = sess.run([train_op, l, merged], feed_dict=feed_dict)

        # Write loss summary
        train_writer.add_summary(summary, global_step=step)

        # Optionally print every 200 steps
        if (step + 1) % 200 == 0:
            print(f"Step {step + 1}/{num_epochs}, Loss: {loss_val:.4f}")

    # Get final weights, bias, probabilities, and predictions
    W_val, b_val, y_prob_val, y_pred_val = sess.run(
        [W, b, y_one_prob, y_pred], feed_dict={x: x_np, y: y_np}
    )


#Plot predicted outputs
plt.figure()
#Color points by predicted class
plt.scatter(x_np[:, 0], x_np[:, 1], c=y_pred_val, cmap='bwr', alpha=0.8)
plt.title('Predicted Classes on Synthetic Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.tight_layout()
plt.show()

#print accuracy and parameters for write-up
accuracy = np.mean(y_pred_val == y_np)
print("Final accuracy on training data:", accuracy)
print("Learned weights W:\n", W_val)
print("Learned bias b:\n", b_val)