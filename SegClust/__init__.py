import warnings

import gc
import numpy as np
import multiprocessing
import cv2

from sklearn.cluster import DBSCAN, Birch
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from SegClust.toolbox import *

warnings.filterwarnings('ignore')


class SegClust:

    def __init__(self):
        self.mode = None
        self.img = None
        self.im_shift = None
        self.im_color_seg = None
        self.xyz = None
        self.xyz_ = None
        self.xyz_birch = None
        self.data_frac = None

        self.xyz_no = None

        self.xyz_centroids = None
        self.fitted_img_xyz = None
        self.fitted_img = None
        self.dist_nn = None

        self.birch_samp_frac = None
        self.dbs_samp_frac = None

        self.img_work_reshape = None
        self.not_outliers = None
        self.dbs_labels = None
        self.divide_labels = None
        self.min_samples = None
        self.min_samples_reduce = None
        self.img_clone = None
        self.img_work_reshape_ = None
        self.img_clone_reshape = None
        self.img_original_reshape = None
        self.img_original_resize_reshape = None
        self.resize_ratio = None
        self.size_limit = None
        self.sp = None
        self.sr = None
        self.blur_size = None

        self.no_pca = False
        self.whiten = False
        self.eps_filter_whiten = 0.1
        self.eps_filter_no_whiten = 0.5

        self.eps_filter = -1.0
        self.birch_thr = 0
        self.stsc = None
        self.pc = None

        self.reconstruction_err = None
        self.model_ratio = None

        self.conf = None
        self.do_prep = True
        self.k_sample = None
        self.n_neigbouhrs = -1

    def set(self, sp=None, sr=60, blur_size=3, sampling_rad=None, min_samples_reduce=100, min_samples=50, size_limit=100,
            n_neigbouhrs=11, whiten=False, no_pca=True, mode='rec'):
        self.__init__()
        self.mode = mode
        self.size_limit = size_limit
        self.sp = sp
        self.sr = sr
        if blur_size == 0:
            self.blur_size = None
        else:
            self.blur_size = blur_size
        self.min_samples = min_samples
        self.min_samples_reduce = min_samples_reduce
        self.do_prep = True
        self.n_neigbouhrs = n_neigbouhrs
        self.whiten = whiten
        self.no_pca = no_pca
        if sampling_rad is not None:
            self.eps_filter = sampling_rad
        elif not no_pca:
            if self.whiten:
                self.eps_filter = self.eps_filter_whiten
            else:
                self.eps_filter = self.eps_filter_no_whiten
        else:
            self.eps_filter = self.eps_filter_no_whiten

    def birch_sample(self):
        self.birch_thr = self.eps_filter/10.
        brc = Birch(branching_factor=50, n_clusters=None, threshold=self.birch_thr, compute_labels=True)
        self.divide_labels = brc.fit_predict(self.xyz)
        tmp_brc = brc.subcluster_centers_
        _frac = 100.*brc.subcluster_centers_.shape[0]/np.float64((self.img_original_reshape.shape[0]*self.img_original_reshape.shape[1]))
        lab_out = np.ones(brc.subcluster_centers_.shape[0], dtype=np.int32)
        agal = DBSCAN(eps=self.eps_filter, min_samples=self.min_samples_reduce, algorithm='ball_tree', n_jobs=-1)
        lab_out = agal.fit_predict(tmp_brc).astype(np.int32)
        _frac = 100.*np.sum(lab_out > -1)/np.float64((self.img_original_reshape.shape[0]*self.img_original_reshape.shape[1]))
        self.dbs_samp_frac = _frac
        return lab_out, tmp_brc

    def prep(self, image):

        self.do_prep = False
        if len(image.shape) < 2:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        self.img_original_reshape = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
        if self.sp is None:
            self.sp = np.int32(np.max(image.shape) / 50.)
        if self.blur_size is not None:
            self.img_clone = cv2.blur(image, (self.blur_size, self.blur_size))
        else:
            self.img_clone = image
        if self.sp > 0 and self.sr > 0:
            self.img_clone = cv2.pyrMeanShiftFiltering(self.img_clone, self.sp, self.sr)
        self.img_clone_reshape = self.img_clone.reshape((self.img_clone.shape[0] * self.img_clone.shape[1], self.img_clone.shape[2]))

        if self.size_limit is not None and np.any(np.array(image.shape) > self.size_limit):
            self.resize_ratio = np.float64(self.size_limit)/np.max(image.shape)
            self.img = cv2.resize(self.img_clone, None, fx=self.resize_ratio, fy=self.resize_ratio)
            im_resize = cv2.resize(image, None, fx=self.resize_ratio, fy=self.resize_ratio)
            self.img_original_resize_reshape = im_resize.reshape((im_resize.shape[0] * im_resize.shape[1], im_resize.shape[2])).copy()
            del im_resize
        else:
            self.img = self.img_clone
            self.img_original_resize_reshape = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))

        # alias allows future preprocessing
        self.im_shift = self.img
        self.img_work_reshape_ = self.im_shift.reshape((self.im_shift.shape[0] * self.im_shift.shape[1], self.im_shift.shape[2])).copy()

        self.xyz_ = self.im_shift.reshape((self.im_shift.shape[0]*self.im_shift.shape[1], self.im_shift.shape[2]))
        self.stsc = StandardScaler(copy=True, with_mean=True, with_std=False)
        self.xyz = self.stsc.fit_transform(self.xyz_)
        self.img_work_reshape = self.stsc.transform(self.img_work_reshape_)
        self.img_clone_reshape = self.stsc.transform(self.img_clone_reshape)
        self.img_original_reshape = self.stsc.transform(self.img_original_reshape)

        if not self.no_pca:
            self.pc = PCA(n_components=None, whiten=self.whiten, copy=True)
            self.xyz = self.pc.fit_transform(self.xyz)
            self.img_work_reshape = self.pc.transform(self.img_work_reshape)
            self.img_clone_reshape = self.pc.transform(self.img_clone_reshape)
            self.img_original_reshape = self.pc.transform(self.img_original_reshape)

        self.dbs_labels, self.xyz_birch = self.birch_sample()
        self.not_outliers = self.dbs_labels > -1

        sampling_frac = 100. * np.sum(self.not_outliers, dtype=np.float64) / (self.img_original_reshape.shape[0])
        self.data_frac = sampling_frac
        self.xyz_no = self.xyz_birch[self.not_outliers, :]
        centroids = self.xyz_no
        self.xyz_centroids_backup = centroids.copy()
        gc.collect()

    def __im_rec_error_(self):
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(self.xyz_centroids)
        dist, idx = nn.kneighbors(X=self.xyz, return_distance=True)
        fitted_img_xyz_train = np.array([self.xyz_centroids[r, :].mean(axis=0) for r in idx])
        err_v_tr = np.sum((self.xyz - fitted_img_xyz_train) ** 2, axis=1)
        ix = np.argsort(err_v_tr, kind='mergesort')[::-1]
        last_err = err_v_tr.mean()
        idx = ix[:, np.newaxis]
        if 'clust' in self.mode:
            nn = NearestNeighbors(n_neighbors=self.min_samples+3, n_jobs=-1)
            nn.fit(self.xyz)
            dist, idx = nn.kneighbors(X=self.xyz, return_distance=True)
            dist = dist[:, 1:]
            self.dist_nn = dist.mean(axis=0)
            idx = idx[:, 1:]
            idx = idx[ix, :]
        err_v_tr = np.asarray([err_v_tr[r].mean() for r in idx])

        return last_err, err_v_tr, idx

    def closeKolor(self, img, n_iter=5, heatpoints=1, correct_model=True):
        """
        agrega puntos para reducir el error de reconstruccion del algo. de sampleo
        :param img:
        :param n_iter:
        :return:
        """
        if n_iter < 1:
            correct_model = False
        if self.do_prep:
            self.prep(img)
        err_sum_tr = []
        self.xyz_centroids = self.xyz_centroids_backup.copy()
        self.reconstruction_err = []
        self.model_ratio = []

        D = self.xyz_centroids.shape[1]

        if correct_model:
            it = 0
            while it < n_iter:
                last_err, err_v_tr, nn_id = self.__im_rec_error_()
                if it < 1 or (it > 0 and last_err < err_sum_tr[-1]):
                    N = self.xyz_centroids.shape[0]
                    M = N + (heatpoints)
                    if 'clust' in self.mode:
                        M = N + (heatpoints * (self.min_samples + 2))
                    self.reconstruction_err.append(last_err)
                    self.model_ratio.append(self.xyz_centroids.shape[0]/np.float64(self.img_original_reshape.shape[0]))
                    err_sum_tr.append(last_err)
                    _idx = nn_id[:heatpoints,:].flatten()
                    tdata = self.xyz[_idx, :]
                    # print(tdata.shape, _idx.size, self.min_samples, M, N)
                    tmp = np.ndarray(shape=(M, D), dtype=np.float64)
                    tmp[:N, :] = self.xyz_centroids
                    tmp[N:, :] = tdata
                    self.xyz_centroids = tmp
                else:
                    it = n_iter
                it += 1
            del tdata
        else:
            last_err, _, _ = self.__im_rec_error_()
            err_sum_tr.append(last_err)
            self.reconstruction_err.append(last_err)
            self.model_ratio.append(self.xyz_centroids.shape[0]/np.float64(self.img_original_reshape.shape[0]))

        self.reconstruction_err = np.array(self.reconstruction_err)
        self.model_ratio = np.array(self.model_ratio)
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(self.xyz_centroids)
        self.xyz_centroids_backup = self.xyz_centroids.copy()
        # reconstruction for the filtered image
        idx = nn.kneighbors(X=self.img_clone_reshape, return_distance=False).ravel()
        self.fitted_img_xyz = np.asarray([self.xyz_centroids[r, :] for r in idx])
        err_v = np.sum((self.img_clone_reshape - self.fitted_img_xyz)**2, axis=1)

        if not self.no_pca:
            img_k = self.pc.inverse_transform(self.fitted_img_xyz)
        else:
            img_k = self.fitted_img_xyz
        img_k = self.stsc.inverse_transform(img_k)
        img_k = np.reshape(img_k, (self.img_clone.shape[0], self.img_clone.shape[1], 3)).astype(np.uint8)
        self.fitted_img = img_k
        return img_k

    def ColorModel(self, img, correct_model=True, model_iter=5, heatpoints=1):

        self.im_color_seg = None
        self.labels_no = None
        self.labels_centroids = None
        self.xyz_no = None

        if self.do_prep:
            if correct_model:
                    self.closeKolor(img, n_iter=model_iter, heatpoints=heatpoints)
            else:
                    self.prep(img)
        self.xyz_centroids = self.xyz_centroids_backup.copy()

        return self.xyz_centroids

    def setup(self, img, correct_model=True, model_iter=5, heatpoints=1):
        if self.do_prep:
            if correct_model:
                    self.closeKolor(img, n_iter=model_iter, heatpoints=heatpoints)
            else:
                    self.prep(img)
        self.xyz_centroids = self.xyz_centroids_backup.copy()

    def fit_predict(self, labels, usecolor=False):

        xyz_no = self.xyz_centroids[labels > -1, :]
        ls = labels[labels > -1]
        knn = KNeighborsClassifier(n_neighbors=self.n_neigbouhrs, n_jobs=-1)
        knn.fit(xyz_no, ls)
        L = knn.predict(self.img_clone_reshape)
        uni = np.unique(ls)
        self.im_color_seg = L.reshape((self.img_clone.shape[0], self.img_clone.shape[1]))

        if usecolor:
            list = sns.color_palette('bright', uni.size)
            c_arr = np.ndarray(shape=(uni.size, 3), dtype=np.uint8)
            L_ = np.ndarray(shape=(self.img_clone.shape[0] * self.img_clone.shape[1], 3), dtype=np.uint8)
            for ix, c in enumerate(list):
                c_arr[ix, :] = (np.array(c)*255).astype(np.uint8)
            for ix, c in enumerate(uni):
                L_[L == c, :] = c_arr[ix, :]
            L_ = L_.reshape((self.img_clone.shape[0], self.img_clone.shape[1], 3))
        else:
            L_ = L.reshape((self.img_clone.shape[0], self.img_clone.shape[1]))
        return L_

    def colorizelabel(self, labels):

        uni = np.unique(labels)
        list = sns.color_palette('bright', uni.size+1)
        c_arr = np.ndarray(shape=(uni.size+1, 3), dtype=np.uint8)
        L_ = np.ndarray(shape=(labels.shape[0] * labels.shape[1], 3), dtype=np.uint8)
        for ix, c in enumerate(list):
            c_arr[ix, :] = (np.array(c)*255).astype(np.uint8)
        for ix, c in enumerate(uni):
            L_[labels.ravel() == c, :] = c_arr[ix, :]
        L_ = L_.reshape((labels.shape[0], labels.shape[1], 3))

        return L_
