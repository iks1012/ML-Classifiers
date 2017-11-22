from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

test = [[190,70,43]]


clf1 = DecisionTreeClassifier()
clf1 = clf1.fit(X,Y)
prediction1 = clf1.predict(test)
print("Decision Tree Classifier Prediction: "+prediction1[0])

clf2 = GaussianNB()
clf2 = clf2.fit(X,Y)
prediction2 = clf2.predict(test)
print("GaussianNB Classifier Prediction: "+prediction2[0])

clf3 = KNeighborsClassifier()
clf3 = clf3.fit(X,Y)
prediction3 = clf3.predict(test)
print("KNeighbors Classifier Prediction: "+prediction3[0])

# They all predict the same thing, but these are different classifiers that can be used for any given type of machine learning.