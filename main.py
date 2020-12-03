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

class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    @staticmethod
    def __Classifiers__(name=None):
        # See for reproducibility
        random_state = 42
        
        if name == 'decision_tree':
            return DecisionTreeClassifier(random_state=random_state)
        if name == 'kneighbors':
            return KNeighborsClassifier()
        if name == 'svm':
            return SVC(random_state=random_state)

    def __DecisionTreeClassifier__(self):

        # Parameters for Grid Search
        decision_tree_params = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4]}
        
        # Decision Tree Classifier
        decision_tree = Ensemble.__Classifiers__(name='decision_tree')
        
        # Grid Search initialization
        decision_tree_grid = GridSearchCV(estimator=decision_tree, param_grid=decision_tree_params, cv=10)
        
        # Init Grid Search
        decision_tree_grid.fit(self.x_train, self.y_train)

        # Logging metrics with MLFlow
        mlflow.log_param(f'dt_best_criterion', decision_tree_grid.best_params_['criterion'])
        mlflow.log_param(f'dt_best_max_depth', decision_tree_grid.best_params_['max_depth'])
        mlflow.log_metric(f'dt_train_acc', decision_tree_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'dt_test_acc', decision_tree_grid.score(self.x_test, self.y_test))

    def __KNearestNeighborsClassifier__(self):

        # Parameters for Grid Search
        knn_params = {'n_neighbors': [3, 5, 10], 'algorithm': ['ball_tree', 'kd_tree', 'brute']}
        
        # K-Nearest Neighbors Classifier
        knn = Ensemble.__Classifiers__(name='kneighbors')
        
        # Grid Search initialization
        knn_grid = GridSearchCV(estimator=knn, param_grid=knn_params, cv=10)
        
        # Init Grid Search
        knn_grid.fit(self.x_train, self.y_train)

        # Logging metrics with MLFlow
        mlflow.log_param(f'knn_best_n_neighbors', knn_grid.best_params_['n_neighbors'])
        mlflow.log_param(f'knn_best_algorithm', knn_grid.best_params_['algorithm'])
        mlflow.log_metric(f'knn_train_acc', knn_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'knn_test_acc', knn_grid.score(self.x_test, self.y_test))

    def __SupporVectorMachineClassifier__(self):

        # Parameters for Grid Search
        svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        
        # Suppor Vector Machine Classifier
        svm = Ensemble.__Classifiers__(name='svm')
        
        # Grid Search initialization
        svm_grid = GridSearchCV(estimator=svm, param_grid=svm_params, cv=10)
        
        # Init Grid Search
        svm_grid.fit(self.x_train, self.y_train)

        # Loggin metrics with MLFlow
        mlflow.log_param(f'svm_best_kernel', svm_grid.best_params_['kernel'])
        mlflow.log_metric(f'svm_train_acc', svm_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'svm_test_acc', svm_grid.score(self.x_test, self.y_test))
    
    def __VotingClassifier__(self):

        # Parameters for Grid Search
        # Notice that it is used as prefix the identifier of each ML model followed by a double underscore
        # e.g. decision_tree__, knn__, svm___
        vc_params = {'decision_tree__criterion': ['gini', 'entropy'], 'decision_tree__max_depth': [2, 3, 4],
                    'knn__n_neighbors': [3, 5, 10], 'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

        # Instantiate classifiers
        decision_tree = Ensemble.__Classifiers__(name='decision_tree')
        knn = Ensemble.__Classifiers__(name='kneighbors')
        svm = Ensemble.__Classifiers__(name='svm')

        # Voting Classifier initialization
        vc = VotingClassifier(estimators=[('decision_tree', decision_tree), ('knn', knn), ('svm', svm)], voting='hard')
        
        # Grid Search initialization
        vc_grid = GridSearchCV(estimator=vc, param_grid=vc_params, cv=10)
        
        # Init Grid Search
        svc_grid.fit(self.x_train, self.y_train)

        # Loggin metrics with MLFlow
        mlflow.log_param(f'vc_best_params', vc_grid.best_params_)
        mlflow.log_metric(f'vc_train_acc', vc_grid.score(self.x_train, self.y_train))
        mlflow.log_metric(f'vc_test_acc', vc_grid.score(self.x_test, self.y_test))

    def __StackingClassifier__(self):
        
        # Instantiate classifiers
        decision_tree = Ensemble.__Classifiers__(name='decision_tree')
        knn = Ensemble.__Classifiers__(name='kneighbors')
        svm = Ensemble.__Classifiers__(name='svm')
        
        # Definition of classifiers base
        estimators = [('decision_tree', decision_tree), ('svm', svm)]
        
        # Stacked Classifier initialization
        # It is defined as final estimator the K-nearest neighbors classifier
        sc = StackingClassifier(estimators=estimators, final_estimator=knn)
        
        # Initi model
        sc.fit(self.x_train, self.y_train)

        # Loggin metrics with MLFlow
        mlflow.log_metric(f'sc_train_acc', sc.score(self.x_train, self.y_train))
        mlflow.log_metric(f'sc_test_acc', sc.score(self.x_test, self.y_test))

if __name__ == "__main__":
    client = MlflowClient()
    experiment_id = client.create_experiment('Ensemble_full_3')

    ensemble = Ensemble()
    ensemble.load_data()

    with mlflow.start_run(experiment_id=experiment_id, run_name='Decision_Tree'):
        ensemble.__DecisionTreeClassifier__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='SVM'):
        ensemble.__SupporVectorMachineClassifier__()

    with mlflow.start_run(experiment_id=experiment_id, run_name='KNN'):
        ensemble.__KNearestNeighborsClassifier__()
    
    # with mlflow.start_run(experiment_id=experiment_id, run_name='test_VC'):
    #     ensemble.__VotingClassifier__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='Stacked_Classifier'):
        ensemble.__StackingClassifier__()