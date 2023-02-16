from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

clf = KNeighborsClassifier()

param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

rs = RandomizedSearchCV(clf, param_grid, cv=5)

rs.fit(X_train, y_train)

best_params = rs.best_params_

clf = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])

cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

print("Cross-validation accuracy: {:.2f}% (+/- {:.2f}%)".format(cv_scores.mean()*100, cv_scores.std()*100))

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100
conf_matrix = confusion_matrix(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)*100
specificity = (conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1]))*100

print('Accuracy:', accuracy,'%')
print('Specificity:', specificity,'%')
print('Sensitivity:', sensitivity,'%')
