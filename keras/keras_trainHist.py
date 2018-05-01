from array import array
import pickle
import matplotlib.pyplot as plt

# prefix = 'Type_B_Keras_Conv_Add_Layer_Dynamic/'
# prefix = 'Type_B_Keras_Conv/'
prefix = 'Type_GPD_Keras_Conv/'

# Training analysis
f_in = open(prefix+'history/training_gpd_alpha0.05_epochs1500_thresh0.02.pkl', 'rb')
training = pickle.load(f_in)

count = [n for n in range(len(training[0]))]
# Plot loss history
fig = plt.figure()
plt.plot(count, training[0], '-')
fig.suptitle("")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(prefix+"perf/loss_history.pdf")
plt.clf()

# Plot difference history
fig = plt.figure()
plt.plot(count, training[1], '-')
# Add markers where an update occurs
markers_x = [c for c, a in zip(count, training[2]) if a == 1]
markers_y = [d for d, a in zip(training[1], training[2]) if a == 1]
print('Number of model additions: %s' % len(markers_x))
plt.plot(markers_x, markers_y, 'g^')
fig.suptitle("")
plt.xlabel("Epoch")
plt.ylabel("Iteration difference")
plt.yscale('log')
plt.savefig(prefix+"perf/diff_history.pdf")
plt.clf()
