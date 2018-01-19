import autograd.numpy as np
from autograd import elementwise_grad, value_and_grad
from optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

f  = lambda x, y: (x-2)**2 + 2*(y-1)**2

ymin, ymax, ystep = -8.5, 8.5, .2

def make_minimize_cb(path=[]):

    def minimize_cb(xk):
        # note that we make a deep copy of xk
        path.append(np.copy(xk))

    return minimize_cb

minima = np.array([2., 1.])

minima_ = minima.reshape(-1, 1)

x0 = np.array([3.0, 2.])

func = value_and_grad(lambda args: f(*args))

methods = ["ALECO2"]

minimize_ = partial(minimize, fun=func, x0=x0, jac=True, bounds=[(xmin, xmax), (ymin, ymax)], tol=1e-20)

paths_ = defaultdict(list)
for method in methods:
    paths_[method].append(x0)

results = {method: minimize_(method=method, callback=make_minimize_cb(paths_[method])) for method in methods}


print (results)
print ()

print ("Solution found by %s:%s"  %(method,results[method]['x']))
print ("True minimum: %s"  %minima)


paths = [np.array(paths_[method]).T for method in methods]
