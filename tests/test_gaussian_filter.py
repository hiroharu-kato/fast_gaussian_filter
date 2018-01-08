import numpy as np
import scipy.misc
import time
import unittest

import chainer
import chainer.gradient_check

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
        print 'diff_in', np.square(diff).sum()
        diff = image_ref - image_out.data.get()
        print 'diff_out', np.square(diff).sum()

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


if __name__ == '__main__':
    unittest.main()
