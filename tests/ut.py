import numpy as np
from models.dynamic_regression import DynamicRegression
from models.gradient_descent import LineSearchTool

def test_predict():
    """
    Unit-tests for predictions.
    """
    # basic
    y_train = np.array([1, 0, 1, 0, 1, 0, 1])
    model = DynamicRegression(fh=np.arange(1, 6))
    model.params = np.array([0, 0.2, 0.2, 0.2, 0.2, 0.2])
    y_test = np.array([0.6, 0.52, 0.624, 0.5488, 0.65856])
    y_preds = model.predict(y_train)
    assert np.allclose(y_test, y_preds), "test_predict basic test failed"

    # trend
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    model = DynamicRegression(fh=np.arange(1, 6))
    model.params = np.array([1, 1, 0, 0, 0, 0])
    y_test = np.array([9, 10, 11, 12, 13])
    y_preds = model.predict(y_train)
    assert np.allclose(y_test, y_preds), "test_predict trend test failed"

    # fh=[5]
    y_train = np.array([1, 0, 1, 0, 1, 0, 1])
    model = DynamicRegression(fh=[5])
    model.params = np.array([0, 0.2, 0.2, 0.2, 0.2, 0.2])
    y_test = np.array([0.65856])
    y_preds = model.predict(y_train)
    assert np.allclose(y_test, y_preds), "test_predict fh=[5] test failed"

    # seasonal
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    model = DynamicRegression(fh=[1, 2], sp=4, ar_depth=2, seas_depth=2)
    model.params = np.array([0, 0.25, 0.25, 0.25, 0.25])
    y_test = np.array([5.25, 5.3125])
    y_preds = model.predict(y_train)
    assert np.allclose(y_test, y_preds), "test_predict seasonal test failed"


def test_grad():
    # grad for one object
    y_train = np.array([1, 0, 1, 0, 1, 0, 1])
    model = DynamicRegression(fh=np.arange(1, 6))
    model.params = np.array([1, -1, 0, 0, 0, 0])
    y_test = np.array([0, 0, 0, 0, 0])
    grad = model.oracle.calc_grad_for_one_object(model.params, y_train, y_test)
    true_grad = np.array([0, -1.2, 1.2, -1.2, 1.2, -1.2])
    assert np.allclose(grad, true_grad), "test_grad grad for one object test failed"
    
    # grad = zero
    y_train = np.array([1, 0, 1, 0, 1, 0, 1])
    model = DynamicRegression(fh=np.arange(1, 6))
    model.params = np.array([1, -1, 0, 0, 0, 0])
    y_test = np.array([0, 1, 0, 1, 0])
    grad = model.oracle.calc_grad_for_one_object(model.params, y_train, y_test)
    true_grad = np.array([0, 0, 0, 0, 0, 0])
    assert np.allclose(grad, true_grad), "test_grad grad = zero test failed"
    
    # learning_steps != 'all'
    y_train = np.array([1, 0, 1, 0, 1, 0, 1])
    model = DynamicRegression(fh=np.arange(1, 6), learning_steps=[0, 2, 4])
    model.params = np.array([1, -1, 0, 0, 0, 0])
    y_test = np.array([0, 0, 0, 0, 0])
    grad = model.oracle.calc_grad_for_one_object(model.params, y_train, y_test)
    true_grad = np.array([0, 0, 0, 0, 0, 0])
    assert np.allclose(grad, true_grad), "test_grad learning_steps != 'all' test failed"

    # seasonal short
    y_train = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    model = DynamicRegression(fh=[1, 2], sp=4, ar_depth=2, seas_depth=2)
    model.params = np.array([0, 0, 0, 1, 0])
    y_test = np.array([1, 4])
    grad = model.oracle.calc_grad_for_one_object(model.params, y_train, y_test)
    true_grad = np.array([-2, -2, -8, -4, -4])
    assert np.allclose(grad, true_grad), "test_grad seasonal short test failed"

    # seasonal long
    y_train = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    model = DynamicRegression(fh=[1, 2, 3, 4, 5], sp=4, ar_depth=2, seas_depth=2)
    model.params = np.array([0, 0, 0, 1, 0])
    y_test = np.array([1, 4, 3, 2, 2])
    grad = model.oracle.calc_grad_for_one_object(model.params, y_train, y_test)
    true_grad = np.array([-0.8, -1.6, -4, 0.8, 0.8])
    assert np.allclose(grad, true_grad), "test_grad seasonal long test failed"

def test_func():
    # basic
    y = np.array([1, 2, -5, 6, 8, 12, 0, 3, -10, 10, 1])
    model = DynamicRegression(ar_depth=1)
    model.params = np.array([1, 2])
    cv_error = model.oracle.func(model.params, y)
    assert np.allclose([253.5], [cv_error]), "test_func basic test failed"

    # seasonal
    y = np.array([1, 2, -5, 6, 8, 12, 0, 3, -10, 10, 1])
    model = DynamicRegression(sp=4, ar_depth=2, seas_depth=2)
    model.params = np.array([0, 1, -1, 1, -1])
    cv_error = model.oracle.func(model.params, y)
    assert np.allclose([381.6666666666667], [cv_error]), "test_func seasonal test failed"


if __name__ == '__main__':
    test_predict()
    test_func()
    test_grad()
