This mini-project contains hands-on implementations of classic classification techniques applied on synthetic and real-world datasets.
The focus is on understanding how different models behave, 
how to interpret their outputs, and how to visualize model decisions and feature importances.

1. xgboost_iris_classifier.py
    •	Dataset: Iris dataset (sklearn.datasets.load_iris)
    •	Model Used: XGBoostClassifier
    •	Key Features:
            o	Multiclass classification task using XGBoost
            o	Includes performance evaluation (accuracy + classification report)
            o	Visualizes feature importance using xgb.plot_importance()

   
3. bayes_knn_2d_classification.py
    •	Dataset: Custom 2D toy dataset with 3 classes
    •	Models Used:
        o	K-Nearest Neighbors (k=3)
        o	Bayes Classifier using:
            	Histogram estimation
            	Parzen Window estimation
            	Gaussian estimation
    •	Key Features:
        o	Classifies a test point using both KNN and Bayesian methods
        o	Visualizes:
              	Scatter plots of classes and test point
              	Histograms of class distributions
              	Parzen and Gaussian density contours
        o	Outputs mean and covariance matrices for each class
        o	Applies Bayes decision rule assuming equal priors

   
3. mlp_digits_classifier.py
      •	Dataset: Digits dataset (sklearn.datasets.load_digits)
      •	Model Used: MLPClassifier (Neural Network)
      •	Key Features:
          o	Compares different hidden layer configurations (e.g. (32,), (64, 64), etc.)
          o	Evaluates metrics: Accuracy, Precision, Recall
          o	Visualizes confusion matrices for each configuration
