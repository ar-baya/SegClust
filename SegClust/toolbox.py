import warnings
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.stats.mstats import mquantiles
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


def a_dist(xp, datap, s):
    a_v = 1. - np.exp(-0.5*cdist(xp[np.newaxis, :], datap, 'sqeuclidean')/s)
    a = a_v.mean()
    return a


def b_dist(xp, data, d_bool, l, s):

    b_v = np.array([(1. - np.exp(-0.5*cdist(xp[np.newaxis, :], data[value, :], 'sqeuclidean')/s)).mean() for key, value in d_bool.iteritems() if key != l])
    b = b_v.min()
    return b


def a_dist_(xp, datap):
    a_v = cdist(xp[np.newaxis, :], datap)
    a = a_v.mean()
    return a


def b_dist_(xp, data, d_bool, l):

    b_v = np.array([(cdist(xp[np.newaxis, :], data[value, :])).mean() for key, value in d_bool.iteritems() if key != l])
    b = b_v.min()
    return b


def silhouette_score_seg(data, labels, sigma):
    ul = np.unique(labels)
    bool_labels = [labels == l for l in ul]
    sum_bool_labels = [np.sum(s) for s in bool_labels]

    bool_labels_d = dict(zip(ul, bool_labels))
    bool_labels_sum = dict(zip(ul, sum_bool_labels))
    if sigma is not None:
        s = sigma**2
        a_i = [a_dist(x, data[bool_labels_d[l], :], s) for x, l in zip(data, labels)]
        b_i = [b_dist(x, data, bool_labels_d, l, sigma) for x, l in zip(data, labels)]
    else:
        a_i = [a_dist_(x, data[bool_labels_d[l], :]) for x, l in zip(data, labels)]
        b_i = [b_dist_(x, data, bool_labels_d, l) for x, l in zip(data, labels)]

    s = np.array([(b-a)/np.max((a, b)) if bool_labels_sum[l] > 1 else 0 for a, b, l in zip(a_i, b_i, labels)])
    return s, s.mean()


def km_sample(data, ppc):

    k = np.int32(1.*data.shape[0]/ppc)
    if k == 0:
        err_arr = np.array([])
        return err_arr, err_arr.shape[0]

    batch_size = np.min((3*k, data.shape[0]))
    # print(data.shape[0], batch_size, k)
    agal = MiniBatchKMeans(n_clusters=k, init='random', max_iter=200, batch_size=batch_size)
    # agal = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=200, batch_size=batch_size)
    agal.fit(data)
    out = agal.cluster_centers_

    return out, out.shape[0]


def kmeans_sampling(data, ppc, not_outliers, divide_labels):

    # stmod = KFold(n_splits=10)
    # split_ix = stmod.split(data)

    boolean_ = [np.logical_and(div_lab == 1, not_outliers) for div_lab in divide_labels]

    # poolsample = multiprocessing.Pool(processes=4)
    # e_pool = [poolsample.apply_async(km_sample, (data[ix, :], ppc)) for _, ix in split_ix]
    # poolsample.close()
    # poolsample.join()
    # del poolsample
    # sampled_data = [e.get() for e in e_pool]
    err_arr = np.array([])
    sampled_data = [km_sample(data[b, :], ppc) if b.sum() > 0 else (err_arr, err_arr.shape[0]) for b in boolean_]
    _size = 0
    for _, size_data in sampled_data:
        _size += size_data
    s_data = np.ndarray(buffer=np.zeros(_size * data.shape[1]), shape=(_size, data.shape[1]),
                        dtype=np.float64)
    ix_init = 0

    while sampled_data:
        d, l = sampled_data.pop()
        if l > 0:
            ix_end = ix_init + l
            s_data[ix_init:ix_end, :] = d
            ix_init += l

    return s_data


def divide(data, ix, size, dim):
    pivot = mquantiles(data[:, dim], prob=[0.5])
    b = data[:, dim] >= pivot
    if dim == 2:
        lt = np.zeros(size, dtype=np.int32)
        lf = np.zeros(size, dtype=np.int32)
        lt[ix[b]] = 1
        lf[ix[np.logical_not(b)]] = 1
        return [lt, lf]
    l_t_ = divide(data[b, :], ix[b], size, dim + 1)
    l_f_ = divide(data[np.logical_not(b), :], ix[np.logical_not(b)], size, dim + 1)
    return [l_t_, l_f_]


def find_best_k(arr, tol):
    arr_max = np.max(arr)
    acc_tol = np.min((tol, arr_max / 10))
    ix_acc = np.arange(0, arr.size, dtype=np.int32)
    ix_acc = ix_acc[arr >= arr_max - acc_tol]
    i_best = np.min(ix_acc)
    return i_best


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_hits(d1, d2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # print('desc ', type(d1), type(d2))
        matches = flann.knnMatch(d1, d2, k=2)
        count_matches = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                count_matches += 1
    except Exception:
        count_matches = 0
    return count_matches


def detect_keypp(desc_list, desc):
    # print('list ', type(desc_list), type(desc))
    hits = np.array([get_hits(des, desc) for des in desc_list])
    return hits


def par_rf(d, c, train_index, test_index, n_estimators):

    X_train, X_test = d[train_index], d[test_index]
    y_train, y_test = c[train_index], c[test_index]
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    acc = accuracy_score(y_test, y)

    return acc

def par_rf_cross(dtr, ctr, dte, cte,train_index, test_index, n_estimators):

    X_train, X_test = dtr[train_index], dte[test_index]
    y_train, y_test = ctr[train_index], cte[test_index]
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    acc = accuracy_score(y_test, y)

    return acc
