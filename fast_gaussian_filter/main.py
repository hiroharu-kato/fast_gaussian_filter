import string

import chainer

import hash_table


def triu(data):
    xp = chainer.cuda.get_array_module(data)
    if hasattr(xp, 'triu'):
        return xp.triu(data)
    else:
        data = xp.ascontiguousarray(data)
        data = chainer.cuda.elementwise(
            'float32 data, int64 size',
            'float32 out',
            '''
                int row = i / size;
                int column = i % size;
                if (column < row) {
                    out = 0;
                } else {
                    out = data;
                }
            ''',
            'kernel',
        )(data, data.shape[0])
        return data


def sort(data):
    xp = chainer.cuda.get_array_module(data)
    data = xp.copy(data)
    dim = data.shape[-1]
    chainer.cuda.elementwise(
        'int32 j, raw float32 data',
        '',
        string.Template('''
            float data_b[${dim}];
            float* p1;
            float* p2;
            float tmp;

            /* copy from global memory */
            p1 = data_b;
            p2 = &data[j * ${dim}];
            for (int k = 0; k < ${dim}; k++) *p1++ = *p2++;

            /* bubble sort */
            for (int k1 = 0; k1 < ${dim} - 1; k1++) {
                for (int k2 = ${dim} - 1; k1 <= k2; k2--) {
                    if (data_b[k2 + 1] < data_b[k2]) {
                        tmp = data_b[k2 + 1];
                        data_b[k2 + 1] = data_b[k2];
                        data_b[k2] = tmp;
                    }
                }
            }

            /* copy to global memory */
            p1 = &data[j * ${dim}];
            p2 = data_b;
            for (int k = 0; k < ${dim}; k++) *p1++ = *p2++;
        ''').substitute(
            dim=dim,
        ),
        'kernel',
    )(xp.arange(data.size / dim).astype('int32'), data)
    return data


def sort2(data):
    # argsort(argsort(data))
    xp = chainer.cuda.get_array_module(data)
    data = xp.copy(data)
    dim = data.shape[-1]
    indices = xp.zeros_like(data, dtype='int32')
    chainer.cuda.elementwise(
        'int32 j, raw float32 data, raw int32 indices',
        '',
        string.Template('''
            float data_b[${dim}];
            int indices1_b[${dim}];
            int indices2_b[${dim}];
            float* p1_f;
            float* p2_f;
            int* p1_i;
            int* p2_i;
            float tmp_f;
            int tmp_i;

            /* copy from global memory */
            p1_f = data_b;
            p2_f = &data[j * ${dim}];
            for (int k = 0; k < ${dim}; k++) *p1_f++ = *p2_f++;

            /* create indices */
            for (int k = 0; k < ${dim}; k++) {
                indices1_b[k] = k;
                indices2_b[k] = k;
            }

            /* bubble sort */
            for (int k1 = 0; k1 < ${dim} - 1; k1++) {
                for (int k2 = ${dim} - 1; k1 <= k2; k2--) {
                    if (data_b[k2 + 1] < data_b[k2]) {
                        tmp_f = data_b[k2 + 1];
                        data_b[k2 + 1] = data_b[k2];
                        data_b[k2] = tmp_f;

                        indices2_b[indices1_b[k2 + 1]] -= 1;
                        indices2_b[indices1_b[k2]] += 1;

                        tmp_i = indices1_b[k2 + 1];
                        indices1_b[k2 + 1] = indices1_b[k2];
                        indices1_b[k2] = tmp_i;
                    }
                }
            }

            /* copy to global memory */
            p1_f = &data[j * ${dim}];
            p2_f = data_b;
            for (int k = 0; k < ${dim}; k++) *p1_f++ = *p2_f++;
            p1_i = &indices[j * ${dim}];
            p2_i = indices2_b;
            for (int k = 0; k < ${dim}; k++) *p1_i++ = *p2_i++;
        ''').substitute(
            dim=dim,
        ),
        'kernel',
    )(xp.arange(data.size / dim).astype('int32'), data, indices)
    return indices


def get_canonical_simplex(xp, dim):
    canonical = xp.zeros((dim + 1, dim + 1), 'int32')
    for i in range(dim + 1):
        canonical[i, :dim + 1 - i] = i
        canonical[i, dim + 1 - i:] = i - (dim + 1)
    return canonical


def get_projection_matrix(xp, dim):
    # create E in p.5 of [Adam+ 2009]
    e_left_u = xp.concatenate((triu(xp.ones((dim, dim), 'float32')), xp.zeros((1, dim), 'float32')), axis=0)
    e_left_d = xp.concatenate((xp.zeros((1, dim), 'float32'), -xp.diag(xp.arange(dim, dtype='float32') + 1)), axis=0)
    e_left = e_left_u + e_left_d
    e_right = xp.diag(1 / xp.sqrt(xp.arange(1, dim + 1, dtype='float32') * xp.arange(2, dim + 2, dtype='float32')))
    e = xp.dot(e_left, e_right)
    e *= xp.sqrt(2.0 / 3.0) * (dim + 1)
    return e


def compute_rzp_rank(xp, features, dim):
    reminder_zero_points = xp.rint(features / (dim + 1)).astype('int32') * (dim + 1)
    # rank = xp.argsort(xp.argsort(-(features - reminder_zero_points), axis=-1), axis=-1)
    rank = sort2(-(features - reminder_zero_points).astype('float32'))
    sum_rzp = (reminder_zero_points / (dim + 1)).sum(-1)
    rank += sum_rzp[:, :, None]

    reminder_zero_points[rank < 0] += dim + 1
    rank[rank < 0] += dim + 1
    reminder_zero_points[dim + 1 <= rank] -= dim + 1
    rank[dim + 1 <= rank] -= dim + 1

    return reminder_zero_points, rank


def compute_weights(xp, features, reminder_zero_points, dim):
    # create b in p.5 of [Adam+ 2009]
    # y = xp.sort(features - reminder_zero_points, -1)[:, :, ::-1].astype('float32')
    y = sort((features - reminder_zero_points).astype('float32'))[:, :, ::-1].astype('float32')
    b = (y[:, :, :-1] - y[:, :, 1:])[:, :, ::-1] / (dim + 1)
    b = xp.concatenate(((1 - b.sum(-1))[:, :, None], b), axis=-1)

    return b


def compute(features, lattice, forward=True):
    # features: [bs, num_points, dim_features]
    # lattice_indices: [bs, num_points, dim_points]
    # barycentric_weights: [bs, num_points, dim_points]
    # lattice_indices_nx: [num_lattice_points, dim_points]
    xp = chainer.cuda.get_array_module(features)
    bs, num_points, dim_features = features.shape
    if forward:
        features = xp.concatenate((features, xp.ones((bs, num_points, 1), 'float32')), axis=2)
        dim_features += 1
    else:
        pass
    num_lattice_points = lattice.lattice_indices_n1.shape[0]
    dim_points = lattice.lattice_indices.shape[2]

    features = xp.ascontiguousarray(features)
    lattice_indices = xp.ascontiguousarray(lattice.lattice_indices)
    barycentric_weights = xp.ascontiguousarray(lattice.barycentric_weights)

    # splatting
    # lattice: [bs, num_lattice_points, dim_features]
    lattice_features = xp.ascontiguousarray(xp.zeros((bs, num_lattice_points, dim_features), 'float32'))
    chainer.cuda.elementwise(
        'raw float32 features, int32 lattice_index, float32 weight, int32 num_lattice_points, raw float32 lattice',
        '',
        string.Template('''
            int bn = i / (${num_points} * ${dim_points});
            int pn = (i % (${num_points} * ${dim_points})) / ${dim_points};
            float* feature_p = &features[(bn * ${num_points} + pn) * ${dim_features}];
            float* lattice_p = &lattice[(bn * num_lattice_points + lattice_index) * ${dim_features}];
            for (int l = 0; l < ${dim_features}; l++) atomicAdd(lattice_p++, weight * *feature_p++);
        ''').substitute(
            num_points=num_points,
            dim_points=dim_points,
            dim_features=dim_features,
        ),
        'kernel',
    )(features, lattice_indices, barycentric_weights, num_lattice_points, lattice_features)

    # blurring
    if forward:
        order = range(dim_points)
    else:
        order = range(dim_points)[::-1]
    for i in order:
        lattice_features_new = xp.ascontiguousarray(xp.zeros_like(lattice_features))
        lin1 = xp.ascontiguousarray(lattice.lattice_indices_n1[:, i])
        lin2 = xp.ascontiguousarray(lattice.lattice_indices_n2[:, i])
        chainer.cuda.elementwise(
            'int32 lin1, int32 lin2, raw float32 lattice, raw float32 lattice_new',
            '',
            string.Template('''
                float* value_p_init = &lattice_new[i * ${dim_features}];

                float* value_p = value_p_init;
                float* value_t_p = &lattice[i * ${dim_features}];
                for (int l = 0; l < ${dim_features}; l++) atomicAdd(value_p++, 0.5 * *value_t_p++);

                if (0 <= lin1) {
                    float* value_n1_p = &lattice[lin1 * ${dim_features}];
                    value_p = value_p_init;
                    for (int l = 0; l < ${dim_features}; l++) atomicAdd(value_p++, 0.25 * *value_n1_p++);
                }

                if (0 <= lin2) {
                    float* value_n2_p = &lattice[lin2 * ${dim_features}];
                    value_p = value_p_init;
                    for (int l = 0; l < ${dim_features}; l++) atomicAdd(value_p++, 0.25 * *value_n2_p++);
                }
            ''').substitute(
                dim_features=dim_features
            ),
            'kernel',
        )(lin1, lin2, lattice_features, lattice_features_new)
        lattice_features = lattice_features_new

    # slicing
    features_out = xp.zeros_like(features, 'float32')
    chainer.cuda.elementwise(
        'raw float32 features, int32 lattice_index, float32 weight, int32 num_lattice_points, raw float32 lattice',
        '',
        string.Template('''
            int bn = i / (${num_points} * ${dim_points});
            int pn = (i % (${num_points} * ${dim_points})) / ${dim_points};
            float* feature_p = &features[(bn * ${num_points} + pn) * ${dim_features}];
            float* lattice_p = &lattice[(bn * num_lattice_points + lattice_index) * ${dim_features}];
            for (int l = 0; l < ${dim_features}; l++) atomicAdd(feature_p++, weight * *lattice_p++);
        ''').substitute(
            num_points=num_points,
            dim_points=dim_points,
            dim_features=dim_features,
        ),
        'kernel',
    )(features_out, lattice_indices, barycentric_weights, num_lattice_points, lattice_features)
    features = features_out
    if forward:
        features, weights = features[:, :, :-1], features[:, :, -1]
        features = features / weights[:, :, None]
        return features, weights
    else:
        return features, None


class Lattice(object):
    def __init__(self, points, hash_size=2 ** 24):
        """
        Input:
            points: [bs, num_points, dim_points]
        Outputs:
            lattice_indices: [bs, num_points, dim_points + 1]
            barycentric_weights: [bs, num_points, dim_points + 1]
            lattice_indices_n1: [bs, num_latteice_points, dim_points + 1]
            lattice_indices_n2: [bs, num_latteice_points, dim_points + 1]
        """
        xp = chainer.cuda.get_array_module(points)
        bs, num_points, dim = points.shape

        #
        projection_matrix = get_projection_matrix(xp, dim)
        points = xp.dot(points, projection_matrix.transpose())
        reminder_zero_points, rank = compute_rzp_rank(xp, points, dim)
        barycentric_weights = compute_weights(xp, points, reminder_zero_points, dim)
        canonical = get_canonical_simplex(xp, dim)
        lattice_points = canonical[:, rank].transpose((1, 2, 0, 3)) + reminder_zero_points[:, :, None, :]
        lattice_points = lattice_points.reshape((lattice_points.shape[0], -1, dim + 1))
        hash = hash_table.HashMap(lattice_points, hash_size)
        lattice_indices = hash.find(lattice_points).reshape((bs, num_points, dim + 1))
        lattice_indices_n1 = xp.zeros((bs, hash.size, dim + 1), 'int32') - 1
        lattice_indices_n2 = xp.zeros((bs, hash.size, dim + 1), 'int32') - 1

        lattice_list = hash.value_list[:, :hash.size]
        for d in range(dim + 1):
            li = xp.copy(lattice_list) - 1
            li[:, :, d] += dim + 1
            lattice_indices_n1[:, :, d] = hash.find(li)
            li = xp.copy(lattice_list) + 1
            li[:, :, d] -= dim + 1
            lattice_indices_n2[:, :, d] = hash.find(li)

        self.lattice_indices = lattice_indices
        self.barycentric_weights = barycentric_weights
        self.lattice_indices_n1 = lattice_indices_n1
        self.lattice_indices_n2 = lattice_indices_n2

    def compute(self, features, forward=True):
        # features: [bs, num_points, dim_features]
        # lattice_indices: [bs, num_points, dim_points]
        # barycentric_weights: [bs, num_points, dim_points]
        # lattice_indices_nx: [num_lattice_points, dim_points]
        xp = chainer.cuda.get_array_module(features)
        bs, num_points, dim_features = features.shape
        num_lattice_points = self.lattice_indices_n1.shape[1]
        dim_points = self.lattice_indices.shape[2]

        features = xp.ascontiguousarray(features)
        lattice_indices = xp.ascontiguousarray(self.lattice_indices)
        barycentric_weights = xp.ascontiguousarray(self.barycentric_weights)

        # splatting
        # lattice: [bs, num_lattice_points, dim_features]
        lattice_features = xp.ascontiguousarray(xp.zeros((bs, num_lattice_points, dim_features), 'float32'))
        if True:
            chainer.cuda.elementwise(
                'raw float32 features, int32 lattice_index, float32 weight, int32 num_lattice_points, ' +
                'raw float32 lattice',
                '',
                string.Template('''
                    int bn = i / (${num_points} * ${dim_points});
                    int pn = (i % (${num_points} * ${dim_points})) / ${dim_points};
                    float* feature_p = &features[(bn * ${num_points} + pn) * ${dim_features}];
                    float* lattice_p = &lattice[(bn * num_lattice_points + lattice_index) * ${dim_features}];
                    // slow
                    // for (int l = 0; l < ${dim_features}; l++) atomicAdd(lattice_p++, weight * *feature_p++);
                    for (int l = 0; l < ${dim_features}; l++) atomicAdd(&lattice_p[l], weight * feature_p[l]);
                ''').substitute(
                    num_points=num_points,
                    dim_points=dim_points,
                    dim_features=dim_features,
                ),
                'kernel',
            )(features, lattice_indices, barycentric_weights, num_lattice_points, lattice_features)
        else:
            # slow
            loop_indices = xp.arange(lattice_indices.size * dim_features).astype('int32')
            chainer.cuda.elementwise(
                'int32 j, raw float32 features, raw int32 lattice_indices, raw float32 weights, ' +
                'int32 num_lattice_points, raw float32 lattice',
                '',
                string.Template('''
                    int bn = i / (${num_points} * ${dim_points} * ${dim_features});
                    int pn = (i % (${num_points} * ${dim_points} * ${dim_features})) / (${dim_points} * ${dim_features});
                    int dim = j % ${dim_features};
                    int lattice_index = lattice_indices[j / ${dim_features}];
                    float weight = weights[j / ${dim_features}];
                    float* feature_p = &features[(bn * ${num_points} + pn) * ${dim_features} + dim];
                    float* lattice_p = &lattice[(bn * num_lattice_points + lattice_index) * ${dim_features} + dim];
                    atomicAdd(lattice_p, weight * *feature_p);
                ''').substitute(
                    num_points=num_points,
                    dim_points=dim_points,
                    dim_features=dim_features,
                ),
                'kernel',
            )(loop_indices, features, lattice_indices, barycentric_weights, num_lattice_points, lattice_features)

        # blurring
        if forward:
            order = range(dim_points)
        else:
            order = range(dim_points)[::-1]
        for i in order:
            lin1 = xp.ascontiguousarray(self.lattice_indices_n1[:, :, i])
            lin2 = xp.ascontiguousarray(self.lattice_indices_n2[:, :, i])
            if True:
                lattice_features_new = lattice_features * 0.5
                chainer.cuda.elementwise(
                    'int32 lin1, int32 lin2, raw float32 lattice, raw float32 lattice_new, int32 num_lattice_points',
                    '',
                    string.Template('''
                        int bn = i / num_lattice_points;
                        float* value_p_init = &lattice_new[i * ${dim_features}];
                        float* value_p;

                        if (0 <= lin1) {
                            float* value_n1_p = &lattice[(bn * num_lattice_points + lin1) * ${dim_features}];
                            value_p = value_p_init;
                            for (int l = 0; l < ${dim_features}; l++) atomicAdd(value_p++, 0.25 * *value_n1_p++);
                        }

                        if (0 <= lin2) {
                            float* value_n2_p = &lattice[(bn * num_lattice_points + lin2) * ${dim_features}];
                            value_p = value_p_init;
                            for (int l = 0; l < ${dim_features}; l++) atomicAdd(value_p++, 0.25 * *value_n2_p++);
                        }
                    ''').substitute(
                        dim_features=dim_features,
                    ),
                    'kernel',
                )(lin1, lin2, lattice_features, lattice_features_new, num_lattice_points)
                lattice_features = lattice_features_new
            else:
                # slow
                loop_indices = xp.arange(lin1.size * dim_features).astype('int32')
                s1 = chainer.cuda.elementwise(
                    'int32 j, raw int32 lins1, raw int32 lins2, raw float32 lattice, int32 num_lattice_points',
                    'float32 ret',
                    string.Template('''
                        int bn = i / (num_lattice_points * ${dim_features});
                        int li = i / ${dim_features};
                        int dim = i % ${dim_features};
                        int lin1 = lins1[li];
                        int lin2 = lins2[li];

                        ret = 0;
                        if (0 <= lin1) ret += lattice[(bn * num_lattice_points + lin1) * ${dim_features} + dim];
                        if (0 <= lin2) ret += lattice[(bn * num_lattice_points + lin2) * ${dim_features} + dim];
                    ''').substitute(
                        dim_features=dim_features,
                    ),
                    'kernel',
                )(loop_indices, lin1, lin2, lattice_features, num_lattice_points)
                s1 = s1.reshape(lattice_features.shape)
                lattice_features = 0.5 * lattice_features + 0.25 * s1

        # slicing
        features_out = xp.zeros_like(features, 'float32')
        chainer.cuda.elementwise(
            'raw float32 features, int32 lattice_index, float32 weight, int32 num_lattice_points, raw float32 lattice',
            '',
            string.Template('''
                int bn = i / (${num_points} * ${dim_points});
                int pn = (i % (${num_points} * ${dim_points})) / ${dim_points};
                float* feature_p = &features[(bn * ${num_points} + pn) * ${dim_features}];
                float* lattice_p = &lattice[(bn * num_lattice_points + lattice_index) * ${dim_features}];
                for (int l = 0; l < ${dim_features}; l++) atomicAdd(feature_p++, weight * *lattice_p++);
            ''').substitute(
                num_points=num_points,
                dim_points=dim_points,
                dim_features=dim_features,
            ),
            'kernel',
        )(features_out, lattice_indices, barycentric_weights, num_lattice_points, lattice_features)
        return features_out


class FastGaussianFilter(chainer.Function):
    def __init__(self, lattice, normalize=True):
        self.lattice = lattice
        self.weights = None
        self.normalize = normalize

    def forward_gpu(self, inputs):
        features = inputs[0]
        if self.normalize:
            xp = chainer.cuda.get_array_module(features)
            features = xp.concatenate((features, xp.ones((features.shape[0], features.shape[1], 1), 'float32')), axis=2)
            features = self.lattice.compute(features)
            features, weights = features[:, :, :-1], features[:, :, -1]
            features = features / (weights[:, :, None] + 1e-6)
            self.weights = weights
        else:
            features = self.lattice.compute(features)
        return features,

    def backward_gpu(self, inputs, grad_outputs):
        grad_output = grad_outputs[0]
        if self.normalize:
            grad_output = grad_output / self.weights[:, :, None]
        grad_input = self.lattice.compute(grad_output, forward=False)
        return grad_input,

    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError


def fast_gaussian_filter(features, points=None, lattice=None, normalize=True):
    if lattice is None:
        lattice = Lattice(points)
    function = FastGaussianFilter(lattice, normalize)
    features = function(features)
    return features
