import numpy as np
from sklearn import grid_search
class SKLearner():

    def __init__(self, estimator):
        self.estimator = estimator
        self.data = []

    def gridsearch(self, params):
        clf = grid_search.GridSearchCV(self.estimator, params)

        X = []
        U = []
        for x, u in self.data:
            X.append(x)
            U.append(u)
        clf.fit(X, U)
        return clf.best_estimator_, clf.best_params_


    def add(self, x, u):
        x = x.reshape(x.size)
        u = u.reshape(u.size)
        self.data.append((x, u))

    def fit(self):
        X = []
        U = []
        for x, u in self.data:
            X.append(x)
            U.append(u)
        return self.estimator.fit(X, U)

    def score(self):
        X = []
        U = []
        for x, u in self.data:
            X.append(x)
            U.append(u)
        return self.estimator.score(X, U)
    
    def predict(self, x):
        x = [x.reshape(x.size)]
        u = np.array(self.estimator.predict(x))
        u = u.reshape((u.size, 1))
        return u

