import os
import cv2
import argparse
import tensorflow as tf

from predict import predict
from export_network import export_network
from dataset import get_dataset
from networks import NET_CONFIGS


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', required=True, type=str)
    parser.add_argument('--name', default='seg_map', type=str)
    parser.add_argument('--dataset-dir', required=True, type=str)
    parser.add_argument('--save-dir', default='./', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--visualize', action="store_true")

    args = parser.parse_args()
    return args


def train_model(model, X, Y, epochs=15, batch_size=16):
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/home/pbuchal/repositories/Cat-Segmentation/Data/Checkpoints/cp-{epoch:0002d}.ckpt',
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=[cp_callback])

    return model


def main():
    args = parse_arguments()

    X, Y = get_dataset(args.dataset_dir)

    net_builder = NET_CONFIGS[args.net]
    model = net_builder(data_format="channels_last")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002),
                  loss=tf.keras.losses.MeanSquaredError())

    model = train_model(model, X, Y, epochs=args.epochs, batch_size=args.batch_iteration)
    export_network(model, NET_CONFIGS[args.net], name=args.name, iteration=args.epochs, input_channel_count=3)

    if args.visualize:
        for image, gt in zip(X, Y):
            response = predict(model, image)
            cv2.imshow('image', image)
            cv2.imshow('ground_truth', gt)
            cv2.imshow('response', response)
            key = cv2.waitKey(0)
            if key == 27:
                break


if __name__ == '__main__':
    main()
