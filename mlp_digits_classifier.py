from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the digits dataset and extract its features and targets
data = load_digits()
feat, targ = data.data, data.target

# split the data into training and testing subsets
train_x, test_x, train_y, test_y = train_test_split(feat, targ, test_size=0.25, random_state=42)

# create a list to collect results for different configurations
results = []

# specify the hidden layer structures to test
layers = [
    (32,),
    (64,),
    (32, 32),
    (64, 64),
    (128, 64)
]

# create a figure for displaying the confusion matrices
plt.figure(figsize=(15, 6))

# initialize an index for the loop
idx = 0

# use a while loop to iterate through the hidden layer configurations
while idx < len(layers):
    config = layers[idx]

    # initialize and train the neural network
    model = MLPClassifier(hidden_layer_sizes=config, max_iter=500, random_state=42)
    model.fit(train_x, train_y)

    # generate predictions on the test set
    preds = model.predict(test_x)

    # calculate performance metrics for the current configuration
    acc = accuracy_score(test_y, preds)
    prec = precision_score(test_y, preds, average='weighted')
    rec = recall_score(test_y, preds, average='weighted')

    # add the calculated metrics to the results list
    results.append({
        'Layers': config,
        'Acc': acc,
        'Prec': prec,
        'Rec': rec
    })

    # compute the confusion matrix for the predictions
    cm = confusion_matrix(test_y, preds)

    # plot the confusion matrix in a subplot
    plt.subplot(1, len(layers), idx + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=data.target_names, yticklabels=data.target_names, square=True)
    plt.title(f'Config: {config}', fontsize=8)
    plt.xlabel('Pred', fontsize=6)
    plt.ylabel('Act', fontsize=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # increment the index for the next iteration
    idx += 1

# organize the layout of the plots and display them
plt.tight_layout()
plt.show()

# transform the results list into a dataframe for analysis
res_df = pd.DataFrame(results)

# print the dataframe containing the performance results
print(res_df)
