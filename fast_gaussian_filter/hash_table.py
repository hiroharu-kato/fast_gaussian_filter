import string

import chainer
import cupy as cp


class HashMap(object):
    def __init__(self, data):
        self.table_size = 2 ** 24
        self.hash_factor = 2531011
        self.dim = data.shape[-1]

        self.indices = cp.ascontiguousarray(cp.zeros((self.table_size,), 'int32')) - 1
        self.values = cp.ascontiguousarray(cp.zeros((self.table_size, self.dim), 'int32'))
        self.value_list = cp.ascontiguousarray(cp.zeros((self.table_size, self.dim), 'int32'))
        self.size = None

        self.init_keys(data)

    def init_keys(self, data):
        data = cp.ascontiguousarray(data)
        used = cp.ascontiguousarray(cp.zeros((self.table_size,), 'int32'))
        written = cp.ascontiguousarray(cp.zeros((self.table_size,), 'int32'))
        count = cp.ascontiguousarray(cp.zeros((1,), 'int32'))
        loop_indices = cp.arange(data.size / self.dim).astype('int32')

        chainer.cuda.elementwise(
            'int32 j, raw int32 data, raw int32 indices, raw int32 values, ' +
            'raw int32 value_list, raw int32 used, raw int32 written, raw int32 count',
            '',
            string.Template('''
                int* value_init;
                int* value;
                value_init = &data[i * ${dim}];

                /* compute initial key */
                unsigned int key = 0;
                value = value_init;
                for (int k = 0; k < ${dim}; k++) key = (key + *value++) * ${hash_factor};
                key = key % ${table_size};

                while (true) {
                    /* check if the key is used */
                    int ret;
                    ret = used[key];
                    if (ret == 0) ret = atomicExch(&used[key], 1);

                    if (ret == 0) {
                        /* register true key */
                        int* value_ref = &values[key * ${dim}];
                        value = value_init;
                        for (int k = 0; k < ${dim}; k++) *value_ref++ = *value++;
                        written[key] = 1;

                        int num = atomicAdd(&count[0], 1);
                        indices[key] = num;

                        value_ref = &value_list[num * ${dim}];
                        value = value_init;
                        for (int k = 0; k < ${dim}; k++) *value_ref++ = *value++;

                        break;
                    } else {
                        bool match = true;
                        while (atomicAdd(&written[key], 0) == 0) {}
                        int* value_ref = &values[key * ${dim}];
                        value = value_init;
                        for (int k = 0; k < ${dim}; k++) if (*value_ref++ != *value++) match = false;
                        if (match) {
                            break;
                        } else {
                            key = (key + 1) % ${table_size};
                        }
                    }
                }
            ''').substitute(
                table_size=self.table_size,
                hash_factor=self.hash_factor,
                dim=self.dim,
            ),
            'kernel',
        )(loop_indices, data, self.indices, self.values, self.value_list, used, written, count)
        self.size = int(count[0])

    def find(self, data):
        ret = cp.ascontiguousarray(cp.zeros(data.shape[:-1], 'int32')) - 1
        data = cp.ascontiguousarray(data)
        loop_indices = cp.arange(data.size / self.dim).astype('int32')
        chainer.cuda.elementwise(
            'int32 j, raw int32 data, raw int32 indices, raw int32 values, raw int32 ret',
            '',
            string.Template('''
                /* */
                int* value = &data[j * ${dim}];

                /* compute initial key */
                unsigned int key = 0;
                for (int k = 0; k < ${dim}; k++) key = (key + value[k]) * ${hash_factor};
                key = key % ${table_size};

                while (1) {
                    if (indices[key] < 0) {
                        ret[j] = -1;
                        break;
                    }
                    bool match = true;
                    for (int k = 0; k < ${dim}; k++) if (values[key * ${dim} + k] != value[k]) match = false;
                    if (match) {
                        ret[j] = indices[key];
                        break;
                    } else {
                        key = (key + 1) % ${table_size};
                    }
                }
            ''').substitute(
                table_size=self.table_size,
                hash_factor=self.hash_factor,
                dim=self.dim,
            ),
            'function',
        )(loop_indices, data, self.indices, self.values, ret)
        return ret
