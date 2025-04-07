import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt

# load the iris dataset
ids = load_iris()
dat, tar = ids.data, ids.target

# split the data into training and testing sets
trn, tst, trl, tsl = train_test_split(
    dat, tar, test_size=0.2, random_state=42, stratify=tar
)

# initialize the xgboost classifier
mod = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)

# train the model
mod.fit(trn, trl)

# make predictions on the test set
prd = mod.predict(tst)

# evaluate the model's performance
acc = accuracy_score(tsl, prd)
rep = classification_report(tsl, prd, target_names=ids.target_names)

print(f"Accuracy of the Model: {acc:.2f}")
print("Classification Report:")
print(rep)

# visualize feature importance
xgb.plot_importance(mod)
plt.show()
