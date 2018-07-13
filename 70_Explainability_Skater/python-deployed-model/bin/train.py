from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib

if __name__ == '__main__':
    boston = load_boston()
    X = boston.data
    y = boston.target

    model = KNeighborsRegressor()
    model.fit(X, y)

    joblib.dump(model, '../models/model.pkl')
