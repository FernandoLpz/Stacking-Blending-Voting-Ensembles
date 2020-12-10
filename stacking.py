import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class Ensemble:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.k = 5

    def load_data(self):
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=23)
    
    def StackingClassifier(self):

        weak_learners = [('dt', DecisionTreeClassifier()),
                        ('knn', KNeighborsClassifier()),
                        ('rf', RandomForestClassifier()),
                        ('gb', GradientBoostingClassifier()),
                        ('gn', GaussianNB())]
        
        final_learner = LogisticRegression()

        meta_model = None
        test_predictions = None

        for clf_id, clf in weak_learners:
            predictions_clf = self.k_fold_cross_validation(clf)
            test_predictions_clf = self.train_level_0(clf)
            
            # Store predictions for the current classifier
            if isinstance(meta_model, np.ndarray):
                meta_model = np.vstack((meta_model, predictions_clf))
            else:
                meta_model = predictions_clf

            # Store predictions for the current classifier
            if isinstance(test_predictions, np.ndarray):
                test_predictions = np.vstack((test_predictions, test_predictions_clf))
            else:
                test_predictions = test_predictions_clf
        
        # Transpose the meta_model
        meta_model = meta_model.T

        # Transpose test predictions
        test_predictions = test_predictions.T

        print(f"Meta model: {meta_model.shape}")
        print(f"Test predictions: {test_predictions.shape}")
        
        self.train_level_1(final_learner, meta_model, test_predictions)


    def k_fold_cross_validation(self, clf):
        
        predictions_clf = None

        # Number of samples per fold
        batch_size = int(len(self.x_train) / self.k)

        # Stars k-fold cross validation
        for fold in range(self.k):

            # Settings for each batch_size
            if fold == (self.k - 1):
                test = self.x_train[(batch_size * fold):, :]
                batch_start = batch_size * fold
                batch_finish = self.x_train.shape[0]
            else:
                test = self.x_train[(batch_size * fold): (batch_size * (fold + 1)), :]
                batch_start = batch_size * fold
                batch_finish = batch_size * (fold + 1)
            
            # test & training samples for each fold iteration
            fold_x_test = self.x_train[batch_start:batch_finish, :]
            fold_x_train = self.x_train[[index for index in range(self.x_train.shape[0]) if index not in range(batch_start, batch_finish)], :]

            # test & training targets for each fold iteration
            fold_y_test = self.y_train[batch_start:batch_finish]
            fold_y_train = self.y_train[[index for index in range(self.x_train.shape[0]) if index not in range(batch_start, batch_finish)]]

            # Fit current classifier
            clf.fit(fold_x_train, fold_y_train)
            fold_y_pred = clf.predict(fold_x_test)

            # Store predictions for each fold_x_test
            if isinstance(predictions_clf, np.ndarray):
                predictions_clf = np.concatenate((predictions_clf, fold_y_pred))
            else:
                predictions_clf = fold_y_pred

        return predictions_clf

    def train_level_0(self, clf):
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        
        return y_pred

    def train_level_1(self, final_learner, meta_model, test_predictions):
        final_learner.fit(meta_model, self.y_train)
        print(f"Train accuracy: {final_learner.score(meta_model,  self.y_train)}")
        print(f"Test accuracy: {final_learner.score(test_predictions, self.y_test)}")
        



if __name__ == "__main__":
    ensemble = Ensemble()
    ensemble.load_data()
    ensemble.StackingClassifier()