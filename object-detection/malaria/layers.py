import keras
import keras.backend
import keras.engine
import keras.engine.topology


class ROI(keras.engine.topology.Layer):
    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = keras.backend.image_dim_ordering()
        assert self.dim_ordering in {'tf',
                                     'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(ROI, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = keras.backend.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            if self.dim_ordering == 'th':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = keras.backend.cast(x1, 'int32')
                        x2 = keras.backend.cast(x2, 'int32')
                        y1 = keras.backend.cast(y1, 'int32')
                        y2 = keras.backend.cast(y2, 'int32')

                        x2 = x1 + keras.backend.maximum(1, x2 - x1)
                        y2 = y1 + keras.backend.maximum(1, y2 - y1)

                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = keras.backend.reshape(x_crop, new_shape)
                        pooled_val = keras.backend.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = keras.backend.cast(x1, 'int32')
                        x2 = keras.backend.cast(x2, 'int32')
                        y1 = keras.backend.cast(y1, 'int32')
                        y2 = keras.backend.cast(y2, 'int32')

                        x2 = x1 + keras.backend.maximum(1, x2 - x1)
                        y2 = y1 + keras.backend.maximum(1, y2 - y1)

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]
                        x_crop = img[:, y1:y2, x1:x2, :]
                        xm = keras.backend.reshape(x_crop, new_shape)
                        pooled_val = keras.backend.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        final_output = keras.backend.concatenate(outputs, axis=0)
        final_output = keras.backend.reshape(final_output, (
            1, self.num_rois, self.pool_size, self.pool_size,
            self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = keras.backend.permute_dimensions(final_output,
                                                            (0, 1, 4, 2, 3))
        else:
            final_output = keras.backend.permute_dimensions(final_output,
                                                            (0, 1, 2, 3, 4))

        return final_output


class BatchNormalization(keras.engine.topology.Layer):
    def __init__(self, epsilon=1e-3, axis=-1,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):
        self.supports_masking = True
        self.beta_init = keras.initializers.get(beta_init)
        self.gamma_init = keras.initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [keras.engine.InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.running_mean = self.add_weight(shape, initializer='zero',
                                            name='{}_running_mean'.format(
                                                self.name),
                                            trainable=False)
        self.running_std = self.add_weight(shape, initializer='one',
                                           name='{}_running_std'.format(
                                               self.name),
                                           trainable=False)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):

        assert self.built, 'Layer must be built before being called'
        input_shape = keras.backend.int_shape(x)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        if sorted(reduction_axes) == range(keras.backend.ndim(x))[:-1]:
            x_normed_running = keras.backend.batch_normalization(
                x, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # need broadcasting
            broadcast_running_mean = keras.backend.reshape(self.running_mean,
                                                           broadcast_shape)
            broadcast_running_std = keras.backend.reshape(self.running_std,
                                                          broadcast_shape)
            broadcast_beta = keras.backend.reshape(self.beta, broadcast_shape)
            broadcast_gamma = keras.backend.reshape(self.gamma,
                                                    broadcast_shape)
            x_normed_running = keras.backend.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon)

        return x_normed_running

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
            'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None
        }
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))