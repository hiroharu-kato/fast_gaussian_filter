# Chainer Implementation of Fast Gaussian Filter

This is an unofficial code for the paper [Fast High-Dimensional Filtering Using the Permutohedral Lattice \[Adams+ Eurographics 2010\]](https://graphics.stanford.edu/papers/permutohedral/).

This GPU implementation (94 &plusmn; 1 ms on GTX 970) is faster than the authors' CPU implementation (827 &plusmn; 4 ms on Core i5-2400) on the reference image.

This code supports derivative of the output image with respect to the input image for neural networks.

## Installation

```sh
sudo python setup.py install
```

## Example

```python
# parameters for bilateral filtering
std_spatial = 5
std_color = 0.125

# read image
image_in = scipy.misc.imread('./ref/in.jpg').astype('float32') / 255.

# create features for filtering
y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
points[:, :, :2] /= std_spatial
points[:, :, 2:] /= std_color

# for fast_gaussian_filter
points = points.reshape((1, -1, 5)) # [batch size, number of pixels, dim of points]
features = image_in.reshape((1, -1, 3)) # [batch size, number of pixels, dim of image]
points = chainer.cuda.to_gpu(points).astype('float32')
features = chainer.cuda.to_gpu(features).astype('float32')

# filter
image_out = fast_gaussian_filter.fast_gaussian_filter(features, points=points)
```

From left-to-right: input image, output image by our code, output image by the authors' code. Mean squared error is 1.412e-10.
<p float="left">
    <img src ="https://raw.githubusercontent.com/hiroharu-kato/fast_gaussian_filter/master/ref/in.jpg" width="30%">
    <img src ="https://raw.githubusercontent.com/hiroharu-kato/fast_gaussian_filter/master/ref/out.jpg" width="30%">
    <img src ="https://raw.githubusercontent.com/hiroharu-kato/fast_gaussian_filter/master/ref/ref.jpg" width="30%">
</p>
