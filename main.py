from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# TODO: 
# - Use emsamble voting with grid search
# - Use stack generalization with grid search

class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
        self.random_state = 42

    def load_data(self):
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=self.random_state)

    def classifiers(self):
        self.dt = DecisionTreeClassifier(random_state=self.random_state)
        self.knn = KNeighborsClassifier()
        self.svm = SVC(random_state=self.random_state)

        self.vc = VotingClassifier(estimators=[('dt', self.dt), ('knn', self.knn), ('svm', self.svm)], voting='hard')
    
    def grid_search(self):

        print(f"Decision Tree:")
        dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4]}
        self.dt_grid = GridSearchCV(estimator=self.dt, param_grid=dt_params, cv=10)
        self.dt_grid.fit(self.x_train, self.y_train)
        print(f"Train acc: {self.dt_grid.score(self.x_train, self.y_train)}")
        print(f"Test acc: {self.dt_grid.score(self.x_test, self.y_test)}")
        
        print(f"\nK-nearest neighbors:")
        knn_params = {'n_neighbors': [3, 5, 10], 'algorithm': ['ball_tree', 'kd_tree', 'brute']}
        self.knn_grid = GridSearchCV(estimator=self.knn, param_grid=knn_params, cv=10)
        self.knn_grid.fit(self.x_train, self.y_train)
        print(f"Train acc: {self.knn_grid.score(self.x_train, self.y_train)}")
        print(f"Test acc: {self.knn_grid.score(self.x_test, self.y_test)}")

        print(f"\nSupport vector machines:")
        svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        self.svm_grid = GridSearchCV(estimator=self.svm, param_grid=svm_params, cv=10)
        self.svm_grid.fit(self.x_train, self.y_train)
        print(f"Train acc: {self.svm_grid.score(self.x_train, self.y_train)}")
        print(f"Test acc: {self.svm_grid.score(self.x_test, self.y_test)}")

        print(f"\nVoting Classifier")
        vc_params = {'dt__criterion': ['gini', 'entropy'], 'dt__max_depth': [2, 3, 4],
                    'knn__n_neighbors': [3, 5, 10], 'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        self.vc_grid = GridSearchCV(estimator=self.vc, param_grid=vc_params, cv=10)
        self.vc_grid.fit(self.x_train, self.y_train)
        print(f"Train acc: {self.vc_grid.score(self.x_train, self.y_train)}")
        print(f"Test acc: {self.vc_grid.score(self.x_test, self.y_test)}")

if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.load_data()
    ensemble.classifiers()
    ensemble.grid_search()
