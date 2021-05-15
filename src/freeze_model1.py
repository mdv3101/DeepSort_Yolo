# vim: expandtab:ts=4:sw=4
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim


#done------------------------------
def batchNormFun(y, scope=None): 
    if scope is not None:
        return slim.batch_norm(y, scope=scope)
    else:
        nm = tf.get_variable_scope().name
        scope = nm + "/bn"
        return slim.batch_norm(y, scope=scope)



    

#partially done model not changed----------------
def linkCreation(
        incident, network_model_Builder, scope, non_Linearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, if_first=False, activationSummarize=True):



    if if_first is False:
        network_model_ = batchNormFun(incident, scope=scope + "/bn")
        network_model_ = non_Linearity(network_model_)
        if activationSummarize is True:
            tf.summary.histogram(scope+"/activations", network_model_)

    else:
        network_model_ = incident


    network_model__per_block = network_model_
    network_model__post_block = network_model_Builder(network_model__per_block, scope)

    
    dimension_outgoing = network_model__post_block.get_shape().as_list()[-1]

    dimension_incident = network_model__per_block.get_shape().as_list()[-1]



    if dimension_incident == dimension_outgoing:
        network_model_ = incident + network_model__post_block
    else:
        projection_network_model_work = slim.conv2d(
            incident, dimension_outgoing, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network_model_ = projection_network_model_work + network_model__post_block


    return network_model_

# not done bcos of models
def inner_block_creation(
        incident, scope, non_Linearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, activationSummarize=True):

    stride_new = 1
    sp1=incident.get_shape()
    sp = sp1.as_list()[-1]
    
    if increase_dim is True:
        stride_new = 2
        sp =sp* 2
        

    incident = slim.conv2d(
        incident, sp, [3, 3], stride_new, activation_fn=non_Linearity, padding="SAME",
        normalizer_fn=batchNormFun, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if activationSummarize:
        tf.summary.histogram(incident.name + "/activations", incident)

    incident = slim.dropout(incident, keep_prob=0.6)

    incident = slim.conv2d(
        incident, sp, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incident

# not done bcos of only function call
def residual_block(incident, scope, non_Linearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, if_first=False,
                   activationSummarize=True):

    def network_model_Builder(x, s):
        return inner_block_creation(
            x, s, non_Linearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, activationSummarize)

    return linkCreation(
        incident, network_model_Builder, scope, non_Linearity, weights_initializer,
        regularizer, if_first, activationSummarize)



# doing 
def network_model__creation(incident, reuse=None, weight_decay=1e-8):
    

    init_fc_weight = tf.truncated_normal_initializer(stddev=1e-3)
    init_fc_bias = tf.zeros_initializer()
    regularizer_fc = slim.l2_regularizer(weight_decay)


    non_Linearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network_model_ = incident
    network_model_ = slim.conv2d(
        network_model_, 32, [3, 3], stride_new=1, activation_fn=non_Linearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)


    network_model_ = slim.conv2d(
        network_model_, 32, [3, 3], stride_new=1, activation_fn=non_Linearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)

    # NOTE(nwojke): This is missing a padding="SAME" to match the CNN
    # architecture in Table 1 of the paper. Information on how this affects
    # performance on MOT 16 training sequences can be found in
    # issue 10 https://github.com/nwojke/deep_sort/issues/10
    network_model_ = slim.max_pool2d(network_model_, [3, 3], [2, 2], scope="pool1")

    network_model_ = residual_block(
        network_model_, "conv2_1", non_Linearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, if_first=True)
    network_model_ = residual_block(
        network_model_, "conv2_3", non_Linearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False)

    network_model_ = residual_block(
        network_model_, "conv3_1", non_Linearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True)
    network_model_ = residual_block(
        network_model_, "conv3_3", non_Linearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False)

    network_model_ = residual_block(
        network_model_, "conv4_1", non_Linearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True)
    network_model_ = residual_block(
        network_model_, "conv4_3", non_Linearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False)

    feature_dim = network_model_.get_shape().as_list()[-1]
    network_model_ = slim.flatten(network_model_)

    network_model_ = slim.dropout(network_model_, keep_prob=0.6)
    network_model_ = slim.fully_connected(
        network_model_, feature_dim, activation_fn=non_Linearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=regularizer_fc,
        scope="fc1", weights_initializer=init_fc_weight,
        biases_initializer=init_fc_bias)

    model_features = network_model_

    # Features in rows, normalize axis 1.
    model_features = slim.batch_norm(model_features, scope="ball", reuse=reuse)
    feature_norm = tf.sqrt(
        tf.constant(1e-8, tf.float32) +
        tf.reduce_sum(tf.square(model_features), [1], keepdims=True))
    model_features = model_features / feature_norm
    return model_features, None

# doing now
def factoryNetwork(weight_decay=1e-8):

    def factory_fn(image, reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=False):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    model_features, logits = network_model__creation(
                        image, reuse=reuse, weight_decay=weight_decay)
                    return model_features, logits

    return factory_fn


def _preprocess(image):
    image = image[:, :, ::-1]  # BGR to RGB
    return image


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Freeze old model")
    parser.add_argument(
        "--checkpoint_in",
        default="resources/network_model_works/mars-small128.ckpt-68577",
        help="Path to checkpoint file")
    parser.add_argument(
        "--graphdef_out",
        default="resources/network_model_works/mars-small128.pb")
    return parser.parse_args()


def main():
    args = parse_args()

    with tf.Session(graph=tf.Graph()) as session:
        input_var = tf.placeholder(
            tf.uint8, (None, 128, 64, 3), name="images")
        image_var = tf.map_fn(
            lambda x: _preprocess(x), tf.cast(input_var, tf.float32),
            back_prop=False)

        factory_fn = factoryNetwork()
        model_features, _ = factory_fn(image_var, reuse=None)
        model_features = tf.identity(model_features, name="features")

        saver = tf.train.Saver(slim.get_variables_to_restore())
        saver.restore(session, args.checkpoint_in)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            session, tf.get_default_graph().as_graph_def(),
            [model_features.name.split(":")[0]])
        with tf.gfile.GFile(args.graphdef_out, "wb") as file_handle:
            file_handle.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    main()
