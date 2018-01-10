import time
import unittest

import chainer
import chainer.gradient_check
import numpy as np
import scipy.misc

import fast_gaussian_filter


class TestFastGaussianFilter(unittest.TestCase):
    def test_forward(self):
        # reference time: 827 +- 4 ms
        # this gpu computation: 104 +- 1 ms
        std_spatial = 5
        std_color = 0.125
        image_in = scipy.misc.imread('./ref/in.jpg').astype('float32') / 255.
        image_ref = np.array([[float(l2) for l2 in l.strip().split(',')] for l in open('./ref/ref.log').readlines()])
        image_ref = image_ref.reshape(image_in.shape)
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5))
        features = image_in.reshape((1, -1, 3))
        points = chainer.cuda.to_gpu(points).astype('float32')
        features = chainer.cuda.to_gpu(features).astype('float32')

        ts = time.time()
        image_out = fast_gaussian_filter.fast_gaussian_filter(features, points=points)
        te = time.time()
        print 'time (1)', te - ts

        for i in range(10):
            ts = time.time()
            image_out = fast_gaussian_filter.fast_gaussian_filter(features, points=points)
            image_out = image_out.reshape(image_ref.shape)
            te = time.time()
            print 'time (2-%d)' % i, te - ts

        lattice = fast_gaussian_filter.Lattice(points)
        for i in range(10):
            ts = time.time()
            image_out = fast_gaussian_filter.fast_gaussian_filter(features, lattice=lattice)
            image_out = image_out.reshape(image_ref.shape)
            te = time.time()
            print 'time (3-%d)' % i, te - ts

        diff = image_in - image_out.data.get()
        print 'diff_in', np.square(diff).mean()
        diff = image_ref - image_out.data.get()
        print 'diff_out', np.square(diff).mean()

        scipy.misc.toimage(image_out.data.get(), cmin=0, cmax=1).save('./ref/out.jpg')

    def test_backward(self):
        # reference time: 827 +- 4 ms
        std_spatial = 5
        std_color = 0.125
        image_in = scipy.misc.imread('./ref/in.jpg').astype('float32') / 255.
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5))
        features = image_in.reshape((1, -1, 3))
        points = chainer.cuda.to_gpu(points).astype('float32')
        features = chainer.cuda.to_gpu(features).astype('float32')

        gy = np.random.normal(size=features.shape).astype('float32')
        gy = chainer.cuda.to_gpu(gy)
        lattice = fast_gaussian_filter.Lattice(points)
        function = fast_gaussian_filter.FastGaussianFilter(lattice)
        chainer.gradient_check.check_backward(function, features, gy, eps=1e-1, atol=1e-3, rtol=1e-3)

    def test_forward2(self):
        # reference time: 827 +- 4 ms
        # this gpu computation: 104 +- 1 ms
        std_spatial = 5
        std_color = 0.125
        image_in = scipy.misc.imread('./ref/in.jpg').astype('float32') / 255.
        image_ref = np.array([[float(l2) for l2 in l.strip().split(',')] for l in open('./ref/ref.log').readlines()])
        image_ref = image_ref.reshape(image_in.shape)
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5))
        features = image_in.reshape((1, -1, 3))
        points = np.tile(points, (2, 1, 1))
        features = np.tile(features, (2, 1, 1))
        points[0] = 0
        features[0] = 0
        points = chainer.cuda.to_gpu(points).astype('float32')
        features = chainer.cuda.to_gpu(features).astype('float32')

        image_out = fast_gaussian_filter.fast_gaussian_filter(features, points=points)
        image_out = image_out.reshape((2, image_in.shape[0], image_in.shape[1], image_in.shape[2]))[1]

        diff = image_in - image_out.data.get()
        print 'diff_in', np.square(diff).mean()
        diff = image_ref - image_out.data.get()
        print 'diff_out', np.square(diff).mean()

    def test_forward3(self):
        # high-dimension
        dim = 64
        std_spatial = 5
        std_color = 0.125
        image_in = scipy.misc.imread('./ref/in.jpg').astype('float32') / 255.
        image_in = image_in[:256, :256]
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5))
        features = np.random.random(size=(image_in.shape[0], image_in.shape[1], dim))
        features = features.reshape((1, -1, features.shape[-1]))
        points = np.tile(points, (2, 1, 1))
        features = np.tile(features, (2, 1, 1))
        points[0] = 0
        features[0] = 0
        points = chainer.cuda.to_gpu(points).astype('float32')
        features = chainer.cuda.to_gpu(features).astype('float32')

        lattice = fast_gaussian_filter.Lattice(points, hash_size=2 ** 20)
        features_out = fast_gaussian_filter.fast_gaussian_filter(features, lattice=lattice)
        print features_out[0, 0, 0]

        ts = time.time()
        features_out = fast_gaussian_filter.fast_gaussian_filter(features, lattice=lattice)
        print features_out[0, 0, 0]
        te = time.time()
        print dim, te - ts


if __name__ == '__main__':
    unittest.main()
