from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

import mlflow
from mlflow.tracking import MlflowClient

# TODO: 
# - Use emsamble voting with grid search
# - Use stack generalization with grid search

class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
        # See for reproducibility
        self.random_state = 42

    def load_data(self):
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=self.random_state)

    def classifiers(self):
        # Classifiers Initialization
        self.dt = DecisionTreeClassifier(random_state=self.random_state)
        self.knn = KNeighborsClassifier()
        self.svm = SVC(random_state=self.random_state)

    def __DecisionTreeClassifier__(self):

        # Parameters for Grid Search
        dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4]}
        # Grid Search initialization
        self.dt_grid = GridSearchCV(estimator=self.dt, param_grid=dt_params, cv=10)
        # Init Grid Search
        self.dt_grid.fit(self.x_train, self.y_train)

        # Logging metrics with MLFlow
        mlflow.log_param(f'dt_best_criterion', self.dt_grid.best_params_['criterion'])
        mlflow.log_param(f'dt_best_max_depth', self.dt_grid.best_params_['max_depth'])
        mlflow.log_metric(f'dt_train_acc', self.dt_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'dt_test_acc', self.dt_grid.score(self.x_test, self.y_test))

    def __KNearestNeighborsClassifier__(self):

        # Parameters for Grid Search
        knn_params = {'n_neighbors': [3, 5, 10], 'algorithm': ['ball_tree', 'kd_tree', 'brute']}
        # Grid Search initialization
        self.knn_grid = GridSearchCV(estimator=self.knn, param_grid=knn_params, cv=10)
        # Init Grid Search
        self.knn_grid.fit(self.x_train, self.y_train)

        # Logging metrics with MLFlow
        mlflow.log_param(f'knn_best_n_neighbors', self.knn_grid.best_params_['n_neighbors'])
        mlflow.log_param(f'knn_best_algorithm', self.knn_grid.best_params_['algorithm'])
        mlflow.log_metric(f'knn_train_acc', self.knn_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'knn_test_acc', self.knn_grid.score(self.x_test, self.y_test))

    def __SupporVectorMachineClassifier__(self):

        # Parameters for Grid Search
        svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        # Grid Search initialization
        self.svm_grid = GridSearchCV(estimator=self.svm, param_grid=svm_params, cv=10)
        # Init Grid Search
        self.svm_grid.fit(self.x_train, self.y_train)

        # Loggin metrics with MLFlow
        mlflow.log_param(f'svm_best_kernel', self.svm_grid.best_params_['kernel'])
        mlflow.log_metric(f'svm_train_acc', self.svm_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'svm_test_acc', self.svm_grid.score(self.x_test, self.y_test))
    
    def __VotingClassifier__(self):

        # Parameters for Grid Search
        # Notice that it is used as prefix the identifier of each ML model followed by a double underscore
        # e.g. dt__, knn__, svm___
        vc_params = {'dt__criterion': ['gini', 'entropy'], 'dt__max_depth': [2, 3, 4],
                    'knn__n_neighbors': [3, 5, 10], 'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

        # Voting Classifier initialization
        self.vc = VotingClassifier(estimators=[('dt', self.dt), ('knn', self.knn), ('svm', self.svm)], voting='hard')
        # Grid Search initialization
        self.vc_grid = GridSearchCV(estimator=self.vc, param_grid=vc_params, cv=10)
        # Init Grid Search
        self.vc_grid.fit(self.x_train, self.y_train)

        # Loggin metrics with MLFlow
        mlflow.log_param(f'vc_best_params', self.vc_grid.best_params_)
        mlflow.log_metric(f'vc_train_acc', self.vc_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'vc_test_acc', self.vc_grid.score(self.x_test, self.y_test))

    def __StackingClassifier__(self):
        # Definition of classifiers base
        estimators = [('dt', self.dt), ('svm', self.svm)]
        # Stacked Classifier initialization
        # It is defined as final estimator the K-nearest neighbors classifier
        self.sc = StackingClassifier(estimators=estimators, final_estimator=self.knn)
        # Initi model
        self.sc.fit(self.x_train, self.y_train)

        # Loggin metrics with MLFlow
        mlflow.log_metric(f'sc_train_acc', self.sc.score(self.x_train, self.y_train))
        mlflow.log_metric(f'sc_test_acc', self.sc.score(self.x_test, self.y_test))

if __name__ == "__main__":
    client = MlflowClient()
    experiment_id = client.create_experiment('Ensemble_full')

    ensemble = Ensemble()
    ensemble.load_data()
    ensemble.classifiers()

    with mlflow.start_run(experiment_id=experiment_id, run_name='test_DT'):
        ensemble.__DecisionTreeClassifier__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='test_SVM'):
        ensemble.__SupporVectorMachineClassifier__()

    with mlflow.start_run(experiment_id=experiment_id, run_name='test_KNN'):
        ensemble.__KNearestNeighborsClassifier__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='test_VC'):
        ensemble.__VotingClassifier__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='test_SC'):
        ensemble.__StackingClassifier__()