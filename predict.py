import cv2
import argparse
import numpy as np
import tensorflow as tf
from dataset import get_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--visualize', default='dataset', choices=['image', 'dataset'])
    parser.add_argument('--data', required=True, type=str)

    args = parser.parse_args()
    return args


def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def predict(model, X):
    Y = model(Input=tf.constant(X[np.newaxis]))
    Y = np.squeeze(Y)
    Y -= Y.min()
    Y /= Y.max()
    return Y


def main():
    args = parse_arguments()

    with tf.io.gfile.GFile(args.model, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    model = wrap_frozen_graph(graph_def=graph_def,
                              inputs=['Input:0'],
                              outputs=['Identity:0'])

    if args.visualize == 'dataset':
        X, Y = get_dataset(args.data)
        for img, gt in zip(X, Y):
            response = predict(model, img)
            cv2.imshow('image', img)
            cv2.imshow('ground_truth', gt.astype('uint8'))
            cv2.imshow('response', response)
            key = cv2.waitKey(0)
            if key == 27:
                break

    elif args.visualize == 'image':
        img = cv2.imread(args.data, flags=cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (640, 480))
        response = predict(model, img)
        cv2.imshow('image', img)
        cv2.imshow('response', response)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
