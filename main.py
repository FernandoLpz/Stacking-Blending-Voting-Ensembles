from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# TODO: 
# - Use emsamble voting with grid search
# - Use stack generalization with grid search  

def classifiers(x_train, y_train):
    random_state = 1
    dt = DecisionTreeClassifier(random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state)
    lr = LogisticRegression(random_state=random_state)
    gn = GaussianNB()



def load_data():
    # Gets and split data
    x, y = load_breast_cancer(return_X_y=True)
    return train_test_split(x, y, test_size=0.2)


if __name__ == "__main__":
    x_train, x_val, y_train, y_val = load_data()
    classifiers(x_train, y_train)