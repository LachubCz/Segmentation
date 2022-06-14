import os
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from networks import InferenceModel


def freeze_model(source_model, network, name, iteration, input_channel_count, save_dir, data_format="channels_last", input_type=tf.uint8):
    if data_format == 'channels_first':
        model = network(data_format="channels_first")
        model.build(input_shape=(None, None, None, input_channel_count))
        model.set_weights(source_model.get_weights())
    else:
        model = source_model

    model = InferenceModel(model, data_format=data_format)
    model.build(input_shape=(None, None, None, input_channel_count))

    full_model = tf.function(lambda x: model(x, training=False))
    full_model = full_model.get_concrete_function(tf.TensorSpec((None, None, None, input_channel_count), input_type, 'Input'))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    if data_format == 'channels_first':
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./",
                      name=os.path.join(save_dir, f"{name}-{iteration:07d}-{data_format}.pb"),
                      as_text=False)

    return ["Input:0"], [f'Identity:0']


def save_toml(name, head_names, iteration, inputs, outputs, save_dir, data_format='NCHW'):
    head_names_string = ""
    segmentation_output = ""
    class_ids_string = ""
    for i, head in enumerate(head_names):
        head_names_string += f'\'{head}\', '
        segmentation_output += f'true, '
        class_ids_string += f'\'{outputs[i]}\', '

    label_mapping_string = ""
    label_mapping_string += f'[LabelMapper.{head_names[0]}.classes]\n'
    label_mapping_string += '\n'

    toml = ('[Detector]\n'
            'version = 1\n'
            'final_threshold = 0\n'
            'final_nms_overlap_threshold = 0\n'
            '\n'
            f'[DetectorStage1.{name}]\n'
            'version = 2\n'
            'method = \'SEG\'\n'
            f'size_step = 0\n'
            f'scale = {[1.0, 1.0]}\n'
            f'invalidated_border = 0\n'
            f'head_names = [{head_names_string}]\n'
            f'segmentation_output = [{segmentation_output}]\n'
            '\n'
            f'{label_mapping_string}'
            f'[DetectorStage1.{name}.Tensorflow]\n'
            f'model_file = \'./{name}-{iteration:07d}-{data_format}.pb\'\n'
            f'input = \'{inputs[0]}\'\n'
            f'class_ids = [{class_ids_string}]\n'
            f'class_probs = []\n'
            '\n'
            )

    with open(os.path.join(save_dir, f'{name}-{iteration:07d}-{data_format}.toml'), "w") as toml_file:
        print(f"{toml}", file=toml_file)


def export_network(source_model, network, save_dir, name, iteration, input_channel_count=3, head_names=['segmentation']):
    inputs, outputs = freeze_model(source_model, network, name, iteration, input_channel_count, save_dir, data_format="channels_first")
    save_toml(name, head_names, iteration, inputs, outputs, save_dir, data_format='NCHW')

    inputs, outputs = freeze_model(source_model, network, name, iteration, input_channel_count, save_dir, data_format='channels_last')
    save_toml(name, head_names, iteration, inputs, outputs, save_dir, data_format='NHWC')


if __name__ == '__main__':
    from networks import NET_CONFIGS
    net_builder = NET_CONFIGS['SimpleNet']
    model = net_builder(data_format='channels_last')
    model.build(input_shape=(None, None, None, 3))
    export_network(source_model=model,
                   network=NET_CONFIGS['SimpleNet'],
                   save_dir='./',
                   name='seg',
                   iteration=200,
                   input_channel_count=3)
