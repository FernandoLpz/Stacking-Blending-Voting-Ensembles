from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.25, random_state=23)

    @staticmethod
    def __Classifiers__(name=None):
        # See for reproducibility
        random_state = 23
        
        if name == 'decision_tree':
            return DecisionTreeClassifier(random_state=random_state)
        if name == 'kneighbors':
            return KNeighborsClassifier()
        if name == 'logistic_regression':
            return LogisticRegression(random_state=random_state)

    def __DecisionTreeClassifier__(self):
        
        # Decision Tree Classifier
        decision_tree = Ensemble.__Classifiers__(name='decision_tree')
        
        # Train Decision Tree
        decision_tree.fit(self.x_train, self.y_train)

        # Logging metrics with MLFlow
        mlflow.log_metric(f'dt_train_acc', decision_tree.score(self.x_train, self.y_train))
        mlflow.log_metric(f'dt_test_acc', decision_tree.score(self.x_test, self.y_test))

    def __KNearestNeighborsClassifier__(self):
        
        # K-Nearest Neighbors Classifier
        knn = Ensemble.__Classifiers__(name='kneighbors')
        
        # Train K-Nearest Neighbos
        knn.fit(self.x_train, self.y_train)

        # Logging metrics with MLFlow
        mlflow.log_metric(f'knn_train_acc', knn.score(self.x_train, self.y_train))
        mlflow.log_metric(f'knn_test_acc', knn.score(self.x_test, self.y_test))

    def __LogisticRegression__(self):
        
        # Decision Tree Classifier
        logistic_regression = Ensemble.__Classifiers__(name='logistic_regression')
        
        # Init Grid Search
        logistic_regression.fit(self.x_train, self.y_train)

        # Logging metrics with MLFlow
        mlflow.log_metric(f'logistic_regression_train_acc', logistic_regression.score(self.x_train, self.y_train))
        mlflow.log_metric(f'logistic_regression_test_acc', logistic_regression.score(self.x_test, self.y_test))
    
    def __VotingClassifier__(self):

        # Instantiate classifiers
        decision_tree = Ensemble.__Classifiers__(name='decision_tree')
        knn = Ensemble.__Classifiers__(name='kneighbors')
        logistic_regression = Ensemble.__Classifiers__(name='logistic_regression')

        # Voting Classifier initialization
        vc = VotingClassifier(estimators=[('decision_tree', decision_tree), ('knn', knn), ('logistic_regression', logistic_regression)], voting='soft')
        
        # Init Grid Search
        vc.fit(self.x_train, self.y_train)

        # Loggin metrics with MLFlow
        mlflow.log_metric(f'vc_train_acc', vc.score(self.x_train, self.y_train))
        mlflow.log_metric(f'vc_test_acc', vc.score(self.x_test, self.y_test))

    # def __StackingClassifier__(self):
        
    #     # Instantiate classifiers
    #     decision_tree = Ensemble.__Classifiers__(name='decision_tree')
    #     knn = Ensemble.__Classifiers__(name='kneighbors')
    #     svm = Ensemble.__Classifiers__(name='svm')
        
    #     # Definition of classifiers base
    #     estimators = [('decision_tree', decision_tree), ('svm', svm)]
        
    #     # Stacked Classifier initialization
    #     # It is defined as final estimator the K-nearest neighbors classifier
    #     sc = StackingClassifier(estimators=estimators, final_estimator=knn)
        
    #     # Initi model
    #     sc.fit(self.x_train, self.y_train)

    #     # Loggin metrics with MLFlow
    #     mlflow.log_metric(f'sc_train_acc', sc.score(self.x_train, self.y_train))
    #     mlflow.log_metric(f'sc_test_acc', sc.score(self.x_test, self.y_test))

if __name__ == "__main__":
    client = MlflowClient()
    experiment_id = client.create_experiment('Ensemble_basic_3')

    ensemble = Ensemble()
    ensemble.load_data()

    with mlflow.start_run(experiment_id=experiment_id, run_name='Decision_Tree'):
        ensemble.__DecisionTreeClassifier__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='SVM'):
        ensemble.__SupporVectorMachineClassifier__()

    with mlflow.start_run(experiment_id=experiment_id, run_name='KNN'):
        ensemble.__KNearestNeighborsClassifier__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='Logistic_Regression'):
        ensemble.__LogisticRegression__()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='VotingClassifier'):
        ensemble.__VotingClassifier__()
    
    # with mlflow.start_run(experiment_id=experiment_id, run_name='Stacked_Classifier'):
    #     ensemble.__StackingClassifier__()