# from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


fm_folder = "../libfm-1.42.src/"

mae_set = []
mse_set = []

for i in xrange(1, 6):
    y = []
    with open(fm_folder + "test/u{}.test".format(i)) as f:
        for line in f:
            # print line
            y.append(float(line.split('\t')[2]))
    y_pred = []
    with open(fm_folder + "output/out{}".format(i)) as f:
        for line in f:
            y_pred.append(float(line.strip()))
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print "============{}".format(i)
    print "MAE: {}".format(mae)
    print "MSE: {}".format(mse)
    mae_set.append(mae)
    mse_set.append(mse)

print "avg:"
print "MAE: {}".format(np.mean(mae_set))
print "MSE: {}".format(np.mean(mse_set))
