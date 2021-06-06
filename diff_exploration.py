import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from pprint import pprint
from time import time
import logging
from joblib import dump, load

data = np.load('data/diff_features.npy')
X, Y = data[:, :-1], data[:, -1:]

pipeline = Pipeline([
    ('minmax', MinMaxScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('reg', Lasso(alpha=1.0,fit_intercept=False))
])

alphas = [0.01, 0.1, 1.0, 10, 100]
polys = [i for i in range(1,2)]

params = {
    "reg__alpha":alphas,
    "poly__degree": polys
}

if __name__ == '__main__':

    # grid = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1)
    #
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(params)
    # t0 = time()
    # grid.fit(X, Y)
    # print("done in %0.3fs" % (time() - t0))
    # print()
    #
    # print("Best score: %0.3f" % grid.best_score_)
    # print("Best parameters set:")
    # best = grid.best_estimator_
    # best_parameters = grid.best_estimator_.get_params()
    # for param_name in sorted(best_parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # dump(best, 'data/best_linear_regression2_std.joblib')
    pipe: Pipeline = load('data/best_linear_regression2.joblib')
    Xt = pipe.named_steps['minmax'].transform(X)
    x = Xt[:,2]
    y = Xt[:,3]
    z = Xt[:,5]
    w = pipe.predict(X)

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.xlim((0,1))
    plt.ylim((0,1))
    ax.set_zlim((0,1))
    ax.scatter3D(x, y, z, c=w)
    plt.show()
