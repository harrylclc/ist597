# from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from util import getArgMap
from math import sqrt

fm_folder = "../libfm-1.42.src/"
argMap = getArgMap(sys.argv[1:])
SAVE_RESULTS = int(argMap.get('-save', 0))

mae_set = []
rmse_set = []

for i in xrange(1, 6):
    # ground truth
    y = []
    with open(fm_folder + "test/u{}.test".format(i)) as f:
        for line in f:
            y.append(float(line.split('\t')[2]))
    # prediction
    y_pred = []
    with open(fm_folder + "output/out{}".format(i)) as f:
        for line in f:
            y_pred.append(float(line.strip()))
    mae = mean_absolute_error(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    print "============{}".format(i)
    print "MAE: {}".format(mae)
    print "RMSE: {}".format(rmse)
    mae_set.append(mae)
    rmse_set.append(rmse)

print "avg:"
print "MAE: {}".format(np.mean(mae_set))
print "MSE: {}".format(np.mean(rmse_set))
if SAVE_RESULTS:
    with open('../result/fm', 'w') as f:
        f.write('MAE: {}\n'.format(np.mean(mae_set)))
        f.write(str(mae_set) + '\n')
        f.write('RMSE: {}\n'.format(np.mean(rmse_set)))
        f.write(str(rmse_set) + '\n')
