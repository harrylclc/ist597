# collaborative filtering: item-oriented approach
from scipy.sparse import csc_matrix
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from util import getArgMap
import sys
data_folder = "/home/cul226/data/ml-100k/"
# data_folder = "/home/lc/data/ml-100k/"


def construct_ui_matrix(input):
    row = []
    col = []
    data = []
    row2cols = dict()
    cols = set()
    with open(input) as f:
        for line in f:
            s = line.strip().split('\t')
            row.append(int(s[0]) - 1)
            col.append(int(s[1]) - 1)
            data.append(int(s[2]))
            row2cols[int(s[0]) - 1] = row2cols.get(int(s[0]) - 1, set())
            row2cols[int(s[0]) - 1].add(int(s[1]) - 1)
            cols.add(int(s[1]) - 1)
    m = csc_matrix((data, (row, col)))
    return m, row2cols, set(row2cols.keys()), cols


def cosine_sim(i, j):
    dot_prod = m[:, i].dot(m[:, j].T).data
    dot_prod = 0 if len(dot_prod) == 0 else dot_prod[0]
    norm_i = np.linalg.norm(m[:, i].data)
    if norm_i == 0:
        return 0
    norm_j = np.linalg.norm(m[:, j].data)
    if norm_j == 0:
        return 0
    cos = dot_prod / (norm_i * norm_j)
    return cos


def pearson_corr(i, j):
    # find the set of users who rated both item i and j
    ui = m[:, i].indices
    uj = m[:, j].indices
    uij = np.intersect1d(ui, uj)
    if len(uij) == 0:
        return 0
    sij, si, sj = 0, 0, 0
    for u in uij:
        sij += (m[u, i] - mean_val[i]) * (m[u, j] - mean_val[j])
        si += (m[u, i] - mean_val[i]) ** 2
        sj += (m[u, j] - mean_val[j]) ** 2
    if si == 0 or sj == 0:
        return 0
    corr = sij / (sqrt(si*sj))
    # print corr
    return corr


def get_sim(i, j):
    if DIST_FUNC == 0:
        return cosine_sim(i, j)
    elif DIST_FUNC == 1:
        return pearson_corr(i, j)


def aggregation_func(nns, u, k, method=0):
    ruk = 0
    if method == 0:  # similarity weights: center data
        sim_sum = 0
        for nn in nns:
            ruk += nn[1] * (m[u, nn[0]] - mean_val[nn[0]])
            sim_sum += abs(nn[1])
        if sim_sum != 0:
            ruk /= sim_sum
            ruk += mean_val[k]
        else:
            ruk = mean_val[k]
    elif method == 1:  # dont center data
        sim_sum = 0
        for nn in nns:
            ruk += nn[1] * m[u, nn[0]]
            sim_sum += abs(nn[1])
        if sim_sum != 0:
            ruk /= sim_sum
    elif method == 2:  # equal weights
        for nn in nns:
            ruk += nn[1]
        ruk /= len(nns)
    return ruk


def predict(u, k):
    if u not in rows and k not in cols:
        print 'user {} and item {} are not in the training set'.format(u, k)
        return 0
    elif u not in rows:
        print "user {} is not in the training set".format(u)
        return mean_val[k]
    elif k not in cols:
        print "item {} is not in the training set".format(k)
        return m[u, :].data.mean()
    if m[u, k] != 0:
        print "already in the training set"
        return m[u, k]
    # find N nearest neighbors
    neighbors = []
    for c in row2cols[u]:
        max_item = max(c, k)
        min_item = min(c, k)
        if USE_COMPUTED_MATRIX:
            sim = sim_mat[min_item][max_item-min_item-1]
        else:
            flag = False
            if max_item in sim_cache:
                if min_item in sim_cache[max_item]:  # hit cache
                    flag = True
                    sim = sim_cache[max_item][min_item]
            if not flag:  # cache the distance
                sim = get_sim(k, c)
                sim_cache[max_item] = sim_cache.get(max_item, dict())
                sim_cache[max_item][min_item] = sim
        neighbors.append((c, sim))
    sort_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
    nns = sort_neighbors[:min(N, len(sort_neighbors))]
    # aggregation
    ruk = aggregation_func(nns, u, k, method=AGGREGATION_METHOD)
    return ruk


def calc_sim_matrix(k):
    with open("../data/u{}.itemsim_method{}".format(k+1, DIST_FUNC), 'w') as f:
        for i in xrange(m.shape[1]):
            print 'calculating {}/{}'.format(i, m.shape[1])
            vals = []
            for j in xrange(i+1, m.shape[1]):
                if len(m[:, i].data) == 0 or len(m[:, j].data) == 0:
                    sim = 0
                else:
                    sim = get_sim(i, j)
                vals.append(str(sim))
            f.write(' '.join(vals))
            f.write('\n')


def load_matrix(k):
    sim_mat = [[] for i in xrange(m.shape[1])]
    print 'loading precompued sim matrix...'
    with open('../data/u{}.itemsim_method{}'.format(k+1, DIST_FUNC)) as f:
        for i in xrange(m.shape[1]):
            s = f.readline().strip().split()
            for j in xrange(i+1, m.shape[1]):
                v = float(s[j-i-1])
                sim_mat[i].append(v)
            # break
    print 'loading done'
    return sim_mat

# parameters
argMap = getArgMap(sys.argv[1:])
N = int(argMap.get('-n', 10))  # N-nearest neighbors
DIST_FUNC = int(argMap.get('-d', 0))  # 0: cos 1: pearson
AGGREGATION_METHOD = int(argMap.get('-a', 0))  # score aggregation method
PRECALCULATION = int(argMap.get('-p', 0))
CV_ID = int(argMap.get('-id', 0))
USE_COMPUTED_MATRIX = int(argMap.get('-m', 0))
SAVE_RESULTS = int(argMap.get('-save', 0))

if __name__ == "__main__":
    mae_set = []
    rmse_set = []
    for i in xrange(5):
        if PRECALCULATION:
            if i != CV_ID:
                continue
        m, row2cols, rows, cols = construct_ui_matrix(
            data_folder + "u{}.base".format(i + 1))
        # pre-computation
        mean_val = []
        for j in xrange(m.shape[1]):
            mean_v = m[:, j].data.mean() if len(m[:, j].data) > 0 else 0
            mean_val.append(mean_v)
        if PRECALCULATION:
            calc_sim_matrix(i)
        else:
            if USE_COMPUTED_MATRIX:
                sim_mat = load_matrix(i)
            else:
                sim_cache = dict()
                # print predict(1,313)
        # break
            uk = []
            y = []
            with open(data_folder + "u{}.test".format(i + 1)) as f:
                for line in f:
                    s = line.strip().split('\t')
                    uk.append((int(s[0]) - 1, int(s[1]) - 1))
                    y.append(int(s[2]))
            y_pred = []
            for (u, k) in uk:
                print u, k
                r_pred = predict(u, k)
                print r_pred
                y_pred.append(r_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = sqrt(mean_squared_error(y, y_pred))
            print "============{}".format(i)
            print "MAE: {}".format(mae)
            print "RMSE: {}".format(rmse)
            mae_set.append(mae)
            rmse_set.append(rmse)
        # break
    if not PRECALCULATION:
        print "avg:"
        print "MAE: {}".format(np.mean(mae_set))
        print "RMSE: {}".format(np.mean(rmse_set))
        if SAVE_RESULTS:
            with open('../result/cf_item_{}_{}_{}'.format(N, DIST_FUNC, AGGREGATION_METHOD), 'w') as f:
                f.write('MAE: {}\n'.format(np.mean(mae_set)))
                f.write(str(mae_set) + '\n')
                f.write('RMSE: {}\n'.format(np.mean(rmse_set)))
                f.write(str(rmse_set) + '\n')
