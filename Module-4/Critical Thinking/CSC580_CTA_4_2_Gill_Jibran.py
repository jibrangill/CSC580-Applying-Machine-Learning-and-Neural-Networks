import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# disable the eager execution of TF2
tf.compat.v1.disable_eager_execution()

# Generate synthetic data
N = 100
np.random.seed(0) 

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

# close writer
train_writer.close()

accuracy = np.mean(y_pred_val == y_np)
#Plot synthetic data and predicted outputs in a single window
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

#left: original synthetic data (class 0 vs class 1)
ax1.scatter(x_zeros[:, 0], x_zeros[:, 1], label='Class 0', color='blue')
ax1.scatter(x_ones[:, 0], x_ones[:, 1], label='Class 1', color='red')
ax1.set_title('Synthetic Data (Class 0 vs Class 1)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend()
ax1.grid(True)

#right: predicted classes
scatter2 = ax2.scatter(x_np[:, 0], x_np[:, 1], c=y_pred_val, cmap='bwr', alpha=0.8)
ax2.set_title('Predicted Classes on Synthetic Data')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.grid(True)

#manual legend for predicted colors
class0_proxy = ax2.scatter([], [], color='blue', label='Predicted Class 0')
class1_proxy = ax2.scatter([], [], color='red', label='Predicted Class 1')
ax2.legend(handles=[class0_proxy, class1_proxy])

#Print accuracy and learned parameters inside the subplot
stats_text = (
    f"Accuracy: {accuracy:.2f}\n"
    f"W = [{W_val[0][0]:.3f}, {W_val[1][0]:.3f}]\n"
    f"b = {b_val[0]:.3f}"
)

#place text in upper-left area of right subplot
ax2.text(
    0.05, 0.95, stats_text,
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
)

plt.tight_layout()
plt.show()