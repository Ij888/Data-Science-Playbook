from sklearn import tree

#training data
#features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"], [120, "smooth"]]
#labels = ["apple", "apple", "orange", "orange" ]

# as SKL uses real-valued features. As in numbers, change all the strings to integers, int.
features = [[140, 1], [130, 1], [150, 0], [120, 0]]
labels = [1, 1, 0, 0, ]

#classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print (clf.predict([[150, 1]]))
