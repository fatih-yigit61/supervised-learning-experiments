import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KernelDensity

# Training data
cls1 = np.array([[2, 1], [2, 2], [2, 4], [3, 2], [3, 3], [4, 1], [6, 1]])
cls2 = np.array([[1, 5], [2, 5], [3, 5], [3, 6], [4, 5]])
cls3 = np.array([[6, 5], [7, 4], [7, 5], [8, 5], [8, 6], [9, 6]])
test_pt = np.array([2, 4])

# (a) Scatter plot to visualize data points
plt.figure(figsize=(6,4))
plt.scatter(cls1[:, 0], cls1[:, 1], label='Class 1', color='red')
plt.scatter(cls2[:, 0], cls2[:, 1], label='Class 2', color='blue')
plt.scatter(cls3[:, 0], cls3[:, 1], label='Class 3', color='green')
plt.scatter(test_pt[0], test_pt[1], label='Test Point', color='black', marker='x', s=100)
plt.legend()
plt.title("Data Visualization (Part a)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True)
plt.show()

print("Part (a) Answer: The scatter plot has been displayed above.")

# (b) KNN Classification for the test point
train_data = np.vstack((cls1, cls2, cls3))
train_labels = np.array([1] * len(cls1) + [2] * len(cls2) + [3] * len(cls3))

knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(train_data, train_labels)
knn_result = knn_model.predict([test_pt])

print(f"Part (b) Answer: Test point classified as Class {knn_result[0]} using KNN.")

# (c) Bayes Classification with Histogram, Parzen Window, and Gaussian Estimation

# Histogram-based visualization
plt.figure(figsize=(6,4))
plt.hist(cls1[:, 0], bins=6, alpha=0.5, label='Class 1 (Feature X)', color='red')
plt.hist(cls2[:, 0], bins=6, alpha=0.5, label='Class 2 (Feature X)', color='blue')
plt.hist(cls3[:, 0], bins=6, alpha=0.5, label='Class 3 (Feature X)', color='green')
plt.legend()
plt.title("Histogram Representation (Part c)")
plt.show()

print("Part (c) Answer: Histogram visualization has been displayed.")

# Parzen Window Visualization
def parzen_plot(data, bandwidth, label, color):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    xy = np.column_stack([x.ravel(), y.ravel()])
    z = np.exp(kde.score_samples(xy)).reshape(x.shape)
    contour = plt.contour(x, y, z, levels=5, alpha=0.7, colors=color)
    # Legend açıklamasını manuel olarak ekliyoruz
    plt.plot([], [], color=color, label=label)  # Boş bir veriyle renk ve etiketi legend'e ekleme

plt.figure(figsize=(6,4))
parzen_plot(cls1, 0.5, 'Class 1 (Parzen)', 'red')
parzen_plot(cls2, 0.5, 'Class 2 (Parzen)', 'blue')
parzen_plot(cls3, 0.5, 'Class 3 (Parzen)', 'green')
plt.scatter(test_pt[0], test_pt[1], color='black', label='Test Point', marker='x', s=100)
plt.legend()
plt.title("Parzen Window Density Estimation")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True)
plt.show()

# Gaussian Model Visualization
def plot_gaussian(data, label, color):
    mean_vec = np.mean(data, axis=0)
    cov_mat = np.cov(data.T)
    x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    xy = np.stack([x, y], axis=-1)
    diff = xy - mean_vec
    cov_inv = np.linalg.inv(cov_mat)
    z = np.einsum('...i,ij,...j', diff, cov_inv, diff)
    det_cov = np.linalg.det(cov_mat)
    gauss_pdf = np.exp(-0.5 * z) / (2 * np.pi * np.sqrt(det_cov))
    plt.contour(x, y, gauss_pdf, levels=5, alpha=0.7, colors=color)
    # Legend için manuel ekleme
    plt.plot([], [], color=color, label=label)  # Boş bir veriyle renk ve etiketi legend'e ekleme
    return mean_vec, cov_mat


plt.figure(figsize=(6,4))
mean1, cov1 = plot_gaussian(cls1, 'Class 1 (Gaussian)', 'red')
mean2, cov2 = plot_gaussian(cls2, 'Class 2 (Gaussian)', 'blue')
mean3, cov3 = plot_gaussian(cls3, 'Class 3 (Gaussian)', 'green')
plt.scatter(test_pt[0], test_pt[1], color='black', label='Test Point', marker='x', s=100)
plt.legend()
plt.title("Gaussian Density Estimation")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True)
plt.show()

# Print Mean and Covariance Matrices
print("Mean and Covariance Matrices:")
print(f"Class 1: Mean={mean1}, Covariance=\n{cov1}")
print(f"Class 2: Mean={mean2}, Covariance=\n{cov2}")
print(f"Class 3: Mean={mean3}, Covariance=\n{cov3}")

# Bayes Decision Rule
prior_probs = [1/3, 1/3, 1/3]  # Assuming equal priors
gauss_cls1 = plot_gaussian(cls1, 'Class 1 (Gaussian)', 'red')[1]
gauss_cls2 = plot_gaussian(cls2, 'Class 2 (Gaussian)', 'blue')[1]
gauss_cls3 = plot_gaussian(cls3, 'Class 3 (Gaussian)', 'green')[1]
posterior_probs = [gauss_cls1 * prior_probs[0], gauss_cls2 * prior_probs[1], gauss_cls3 * prior_probs[2]]
bayes_result = np.argmax(posterior_probs) + 1

print(f"Part (c) Answer (Bayes Decision): Test point classified as Class {bayes_result}.")

