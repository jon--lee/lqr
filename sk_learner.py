import numpy as np

class SKLearner():

    def __init__(self, estimator):
        self.estimator = estimator
        self.data = []

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

