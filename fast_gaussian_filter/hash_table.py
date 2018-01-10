import string

import chainer
import cupy as cp


class HashMap(object):
    def __init__(self, data, table_size=2 ** 24):
        xp = chainer.cuda.get_array_module(data)
        self.hash_factor = 2531011
        self.batch_size, self.num_points, self.dim = data.shape
        self.table_size = table_size

        self.indices = cp.ascontiguousarray(cp.zeros((self.batch_size, self.table_size,), 'int32')) - 1
        self.values = cp.ascontiguousarray(cp.zeros((self.batch_size, self.table_size, self.dim), 'int32'))
        self.value_list = cp.ascontiguousarray(cp.zeros((self.batch_size, self.table_size, self.dim), 'int32'))
        self.size = None

        self.init_keys(data)

    def init_keys(self, data):
        data = cp.ascontiguousarray(data)
        used = cp.ascontiguousarray(cp.zeros((self.batch_size, self.table_size), 'int32'))
        written = cp.ascontiguousarray(cp.zeros((self.batch_size, self.table_size), 'int32'))
        count = cp.ascontiguousarray(cp.zeros((self.batch_size,), 'int32'))
        ok = cp.zeros((1,), 'int32')
        loop_indices = cp.arange(data.size / self.dim).astype('int32')

        chainer.cuda.elementwise(
            'int32 j, raw int32 data, raw int32 indices, raw int32 values, ' +
            'raw int32 value_list, raw int32 used, raw int32 written, raw int32 count, raw int32 ok',
            '',
            string.Template('''
                int* value_init;
                int* value;
                value_init = &data[i * ${dim}];
                int bn = i / ${num_points};

                /* compute initial key */
                unsigned int key = 0;
                value = value_init;
                for (int k = 0; k < ${dim}; k++) key = (key + *value++) * ${hash_factor};
                key = key % ${table_size};

                for (int l = 0; l < 100; l++) {
                    /* check if the key is used */
                    int ret;
                    ret = used[bn * ${table_size} + key];
                    if (ret == 0) ret = atomicExch(&used[bn * ${table_size} + key], 1);

                    if (ret == 0) {
                        /* register true key */
                        int* value_ref = &values[(bn * ${table_size} + key) * ${dim}];
                        value = value_init;
                        for (int k = 0; k < ${dim}; k++) *value_ref++ = *value++;
                        written[bn * ${table_size} + key] = 1;

                        int num = atomicAdd(&count[bn], 1);
                        indices[bn * ${table_size} + key] = num;

                        value_ref = &value_list[(bn * ${table_size} + num) * ${dim}];
                        value = value_init;
                        for (int k = 0; k < ${dim}; k++) *value_ref++ = *value++;

                        break;
                    } else {
                        bool match = true;
                        while (atomicAdd(&written[bn * ${table_size} + key], 0) == 0) {}
                        int* value_ref = &values[(bn * ${table_size} + key) * ${dim}];
                        value = value_init;
                        for (int k = 0; k < ${dim}; k++) if (*value_ref++ != *value++) match = false;
                        if (match) {
                            break;
                        } else {
                            key = (key + 1) % ${table_size};
                        }
                    }
                    if (l == 99) {
                        ok[0] = -1;
                    }
                }
            ''').substitute(
                table_size=self.table_size,
                hash_factor=self.hash_factor,
                num_points=self.num_points,
                dim=self.dim,
            ),
            'kernel',
        )(loop_indices, data, self.indices, self.values, self.value_list, used, written, count, ok)
        self.size = int(count.max())
        if int(ok[0]) < 0:
            raise Exception

    def find(self, data):
        ret = cp.ascontiguousarray(cp.zeros(data.shape[:-1], 'int32')) - 1
        data = cp.ascontiguousarray(data)
        loop_indices = cp.arange(data.size / self.dim).astype('int32')
        ok = cp.zeros((1,), 'int32')
        chainer.cuda.elementwise(
            'int32 j, raw int32 data, raw int32 indices, raw int32 values, raw int32 ret, raw int32 ok',
            '',
            string.Template('''
                /* */
                int* value = &data[j * ${dim}];
                int bn = i / ${num_points};

                /* compute initial key */
                unsigned int key = 0;
                for (int k = 0; k < ${dim}; k++) key = (key + value[k]) * ${hash_factor};
                key = key % ${table_size};

                for (int l = 0; l < 100; l++) {
                    if (indices[bn * ${table_size} + key] < 0) {
                        ret[j] = -1;
                        break;
                    }
                    bool match = true;
                    for (int k = 0; k < ${dim}; k++)
                        if (values[(bn * ${table_size} + key) * ${dim} + k] != value[k])
                            match = false;
                    if (match) {
                        ret[j] = indices[bn * ${table_size} + key];
                        break;
                    } else {
                        key = (key + 1) % ${table_size};
                    }
                    if (l == 99) {
                        ok[0] = -1;
                    }
                }
            ''').substitute(
                table_size=self.table_size,
                hash_factor=self.hash_factor,
                num_points=data.shape[1],
                dim=self.dim,
            ),
            'function',
        )(loop_indices, data, self.indices, self.values, ret, ok)
        if int(ok[0]) < 0:
            raise Exception
        return ret
