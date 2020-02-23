import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# Select needed data/attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

# Label - What we're looking for
predict = "G3"

# New data frame without the G3 attribute
x = np.array(data.drop([predict], 1))

# Labels data frame
y = np.array(data[predict])

"""
acc = 0
while acc < 0.95:
    # Take all attributes and labels and split into 4 different arrays
    # Test data will also be created to test the accuracy of the model
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    # Find best fit line with the training data
    linear.fit(x_train, y_train)

    # Accuracy of the model
    acc = linear.score(x_test, y_test)
    print(acc)

# Save a pickle file that contains the model in the directory
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)
    
print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
"""

# Load the model from the pickle file
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Plot data p and its correlation to G3
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()