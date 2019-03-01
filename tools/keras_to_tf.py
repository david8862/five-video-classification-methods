#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will convert tf.keras model file to
Tensorflow pb model.
"""
import os
import ast
import argparse

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

K.set_learning_phase(False)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def save_model(keras_model, session, pb_model_path):
    x = keras_model.input
    y = keras_model.output
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"inputs": x}, {"prediction": y})
    builder = tf.saved_model.builder.SavedModelBuilder(pb_model_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    signature = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature, }
    builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature, legacy_init_op=legacy_init_op)
    builder.save()


def run(keras_model_file, output_path):
    sess = K.get_session()
    model = load_model(keras_model_file)
    output_names = [node.op.name for node in model.outputs]
    _ = freeze_session(sess, output_names=output_names)
    save_model(keras_model=model, session=sess, pb_model_path=output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='Full path of the output pb model.', type=str)
    parser.add_argument('--keras_model_file', help='Full filepath of HDF5 file containing tf.Keras model.', type=str)

    args = parser.parse_args()
    if not args.output_path:
        raise ValueError('output_path not specified')
    if not args.keras_model_file:
        raise ValueError('keras_model_file not specified')

    run(args.keras_model_file, args.output_path)


if __name__ == "__main__":
    main()
