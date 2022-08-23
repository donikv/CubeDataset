import tensorflow as tf


def create_mask(size=tf.constant((480, 480)), transpose=tf.constant(False)):
    if transpose:
        start = tf.random.uniform([], 0, size[0], tf.int32)
        line_index = tf.range(size[1], dtype=tf.int32)
        line = tf.zeros((size[1]), dtype=tf.int32)
        x_size = size[0]
        y_size = size[1]
    else:
        start = tf.random.uniform([], 0, size[1], tf.int32)
        line_index = tf.range(size[0], dtype=tf.int32)
        line = tf.zeros((size[0]), dtype=tf.int32)
        x_size = size[1]
        y_size = size[0]

    y_current = -1
    x_current = start

    prob = tf.random.uniform([5], 0, 100, tf.float32)

    def cond(x_size, x_current, y_size, y_current, line, line_index):
        return tf.math.less(y_current, y_size)

    def body(x_size, x_current, y_size, y_current, line, line_index):
        select = tf.constant([0, 1, 2, 3, 4])
        sample = tf.squeeze(tf.random.categorical(tf.math.log([prob]), 1))
        step = select[sample]
        if tf.equal(step, 0):
            x_current -= 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        elif tf.equal(step, 1):
            x_current -= 1
            y_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        elif tf.equal(step, 2):
            y_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        elif tf.equal(step, 3):
            x_current += 1
            y_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        else:
            x_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)

        return x_size, x_current, y_size, y_current, line, line_index

    _, _, _, _, line, _ = tf.while_loop(cond, body, [x_size, x_current, y_size, y_current, line, line_index],
                                        parallel_iterations=1)

    if transpose:
        x = tf.tile([tf.range(0, size[0], dtype=tf.int32)], [size[1], 1])
        rez = x - tf.expand_dims(line, 1)
        rez = tf.transpose(rez)
    else:
        x = tf.tile([tf.range(0, size[1], dtype=tf.int32)], [size[0], 1])
        rez = x - tf.expand_dims(line, 1)

    ones = tf.ones(size)
    zeros = tf.zeros(size) - 1

    rez = tf.where(rez >= 0, ones, zeros)

    return rez