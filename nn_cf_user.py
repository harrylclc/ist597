# collaborative filtering: user-oriented approach

from scipy.sparse import csr_matrix
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
    col2rows = dict()
    rows = set()
    with open(input) as f:
        for line in f:
            s = line.strip().split('\t')
            row.append(int(s[0]) - 1)
            col.append(int(s[1]) - 1)
            data.append(int(s[2]))
            col2rows[int(s[1]) - 1] = col2rows.get(int(s[1]) - 1, set())
            col2rows[int(s[1]) - 1].add(int(s[0]) - 1)
            rows.add(int(s[0]) - 1)
    m = csr_matrix((data, (row, col)))
    return m, col2rows, rows, set(col2rows.keys())


def cosine_sim(u, v):
    # find the set of items rated by both user u and v
    iu = m[u, :].indices
    iv = m[v, :].indices
    iuv = np.intersect1d(iu, iv)
    if len(iuv) == 0:
        return 0
    dot_prod = m[u, iuv].dot(m[v, iuv].T).data[0]
    # dot_prod = 0 if len(dot_prod) == 0 else dot_prod[0]
    norm_x = np.linalg.norm(m[u, iuv].data)
    if norm_x == 0:
        return 0
    norm_y = np.linalg.norm(m[v, iuv].data)
    if norm_y == 0:
        return 0
    cos = dot_prod / (norm_x * norm_y)
    # print cos
    return cos


def pearson_corr(u, v):
    # find the set of items rated by both user u and v
    iu = m[u, :].indices
    iv = m[v, :].indices
    iuv = np.intersect1d(iu, iv)
    if len(iuv) == 0:
        return 0
    suv, su, sv = 0, 0, 0
    for i in iuv:
        suv += (m[u, i] - mean_val[u]) * (m[v, i] - mean_val[v])
        su += (m[u, i] - mean_val[u]) ** 2
        sv += (m[v, i] - mean_val[v]) ** 2
    if su == 0 or sv == 0:
        return 0
    corr = suv / (sqrt(su * sv))
    return corr


def get_sim(u, v):
    if DIST_FUNC == 0:
        return cosine_sim(u, v)
    elif DIST_FUNC == 1:
        return pearson_corr(u, v)


def aggregation_func(nns, u, k, method=0):
    ruk = 0
    if method == 0:
        sim_sum = 0
        for nn in nns:
            ruk += nn[1] * (m[nn[0], k] - mean_val[nn[0]])
            sim_sum += abs(nn[1])
        if sim_sum != 0:
            ruk /= sim_sum
            ruk += mean_val[u]
        else:
            ruk = mean_val[u]
    elif method == 1:
        sim_sum = 0
        for nn in nns:
            ruk += m[nn[0], k] * nn[1]
            sim_sum += abs(nn[1])
        if sim_sum != 0:
            ruk /= sim_sum
    return ruk


def predict(u, k):
    if u not in rows and k not in cols:
        print 'user {} and item {} are not in the training set'.format(u, k)
        return 0
    elif k not in cols:
        print "item {} is not in the training set".format(k)
        return mean_val[u]
    elif u not in rows:
        print 'user {} is not in the training set'.format(u)
        return m[:, k].data.mean()
    if m[u, k] != 0:
        print "already in the training set"
        return m[u, k]
    # find N nearest neighbors
    neighbors = []
    for r in col2rows[k]:
        max_user = max(u, r)
        min_user = min(u, r)
        if USE_COMPUTED_MATRIX:
            sim = sim_mat[min_user][max_user-min_user-1]
        else:
            flag = False
            if max_user in sim_cache:
                if min_user in sim_cache[max_user]:  # hit cache
                    flag = True
                    sim = sim_cache[max_user][min_user]
            if not flag:
                sim = get_sim(u, r)
                sim_cache[max_user] = sim_cache.get(max_user, dict())
                sim_cache[max_user][min_user] = sim
        neighbors.append((r, sim))
    sort_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
    nns = sort_neighbors[:min(N, len(sort_neighbors))]
    # aggregation
    ruk = aggregation_func(nns, u, k, method=AGGREGATION_METHOD)
    return ruk


def calc_sim_matrix(k):
    with open("../data/u{}.usersim_method{}".format(k+1, DIST_FUNC), 'w') as f:
        for i in xrange(m.shape[0]):
            print 'calculating {}/{}'.format(i, m.shape[0])
            vals = []
            for j in xrange(i+1, m.shape[0]):
                if len(m[i, :].data) == 0 or len(m[j, :].data) == 0:
                    sim = 0
                else:
                    sim = get_sim(i, j)
                vals.append(str(sim))
            f.write(' '.join(vals))
            f.write('\n')


def load_matrix(k):
    sim_mat = [[] for i in xrange(m.shape[0])]
    print 'loading precompued sim matrix...'
    with open('../data/u{}.usersim_method{}'.format(k+1, DIST_FUNC)) as f:
        for i in xrange(m.shape[0]):
            s = f.readline().strip().split()
            for j in xrange(i+1, m.shape[0]):
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
USE_COMPUTED_MATRIX = int(argMap.get('-m', 0))
SAVE_RESULTS = int(argMap.get('-save', 0))

if __name__ == "__main__":
    mae_set = []
    rmse_set = []
    for i in xrange(5):
        m, col2rows, rows, cols = construct_ui_matrix(
            data_folder + "u{}.base".format(i + 1))
        # pre-computation
        mean_val = []
        for j in xrange(m.shape[0]):
            mean_val.append(m[j, :].data.mean())
        if PRECALCULATION:
            calc_sim_matrix(i)
        else:
            if USE_COMPUTED_MATRIX:
                sim_mat = load_matrix(i)
            else:
                sim_cache = dict()
            # test
            uk = []
            y = []
            with open(data_folder + "u{}.test".format(i + 1)) as f:
                for line in f:
                    s = line.strip().split('\t')
                    uk.append((int(s[0]) - 1, int(s[1]) - 1))
                    y.append(int(s[2]))
            y_pred = []
            # predict(6, 598)
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
        with open('../result/cf_user_{}_{}_{}'.format(N, DIST_FUNC, AGGREGATION_METHOD), 'w') as f:
                f.write('MAE: {}\n'.format(np.mean(mae_set)))
                f.write(str(mae_set) + '\n')
                f.write('RMSE: {}\n'.format(np.mean(rmse_set)))
                f.write(str(rmse_set) + '\n')
