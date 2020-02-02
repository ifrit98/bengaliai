
import numpy as np
import tensorflow as tf
import json
import os
import glob
from tempfile import NamedTemporaryFile
# tf.enable_eager_execution()

JSON_FNAME = "FEATURE_DATA.json"

def sink_generator(batch_generator, output_dir, num_batches,
                   batches_per_file=25):
    """
    Parameters
    batch_generator: instance of a batch generator, must return a dict
    output_dir: string, path to output directory
    num_batches: int, number of batches to sink
    batches_per_file: int, number of batches per tf record

    Will write a json named 'FEATURE_DATA.json' to help read the tfrecords later
    using the function 'read_dataset'.
    Will also write 'ceil(num_batches / batches_per_file)' tfrecords

    Usage example:
    >>> bg = my_batch_generator(batch_size=32)
    >>> print(type(bg.next()))
        'dict'
    >>> sink_generator(bg, '/path/to/tfrecords/', 50)
    >>> files = os.listdir('/path/to/tfrecords/')
    >>> for f in files:
    >>>     print(f)
        'FEATURE_DICT.json'
        'tmpaqi23x.tfrecord'
        'tmpu9cbl2.tfrecord'
    """
    output_dir = os.path.abspath(output_dir) + '/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # --------check if json feature dict already exists in directory-----------
    json_files = glob.glob(output_dir + "*.json") # remove

    if len(json_files) not in [0, 1]:
        raise ValueError('More than one json file already in directory')

    if len(json_files) == 1:
        json_file_exists = True
        with open(json_files[0]) as f:
            json_dict = json.load(f)
    else:
        json_file_exists = False

    for i in range(0, num_batches, batches_per_file):
        output_record = NamedTemporaryFile(dir=output_dir,
                                           suffix=".tfrecord", delete=False)
        writer = tf.python_io.TFRecordWriter(output_record.name)

        for _ in range(min(num_batches-i, batches_per_file)):
            if json_file_exists:
                batch = batch_generator.next()
                if _is_normal_batch(batch, json_dict):
                    _write_to_tf_record(batch, writer)
            else:
                batch = batch_generator.next()
                assert isinstance(batch, dict)
                output_json_file = open(os.path.join(output_dir, 'FEATURE_DATA.json'), 'w')
                json_dict = _build_json_dict(batch)
                json.dump(json_dict, output_json_file)
                json_file_exists = True
                output_json_file.close()

                _write_to_tf_record(batch, writer)

        output_record.close()


# DEPRECATED
def read_tf_record(file_prefix):
    """
    Parameters:
    file_prefix: string, path to tfrecord to read excluding '.tfrecord'
                e.g. '/path/to/tfrecords/tmpdf3rcm.tfrecord' -> 
                      '/path/to/tfrecords/'tmpdf3rcm'

    Returns:
    tf.data.TFRecordDataset object that can iterate through the tfrecord
    returning samples as tensors


    Usage example:
    >>> my_dataset = read_tf_record('/path/to/tfrecords/tmpaqi23x')
    >>> iterator = my_dataset.make_one_shot_iterator()
    >>> sample_tensor = iterator.get_next()
    >>> sess = tf.Session()
    >>> sample = sess.run(sample_tensor)
    >>> print(sample['baud'])
        3102.34098

    """

    # ---------build feature dictionary from json-----------------------
    READ_FNCS = {
        'FixedLen': tf.FixedLenFeature,
        'FixedLenSequence': tf.FixedLenSequenceFeature
    }
    STR_TO_DTYPE = {
        'tf.float32': tf.float32,
        'tf.int64': tf.int64,
        'tf.string': tf.string
    }

    with open(file_prefix + '.json') as json_file:
        feature_dicts = json.load(json_file)

    features = {}
    for field_key in feature_dicts.keys():

        # remove odd 'u' from keys as a result of json
        field_key = str(field_key)
        for sub_key in feature_dicts[field_key].keys():
            sub_key = str(sub_key)

        fnc_str = feature_dicts[field_key]['fnc']
        dtype_str = feature_dicts[field_key]['dtype']
        shape = feature_dicts[field_key]['shape']

        fnc = READ_FNCS[fnc_str]
        dtype = STR_TO_DTYPE[dtype_str]
        if fnc.__name__ == 'FixedLenSequenceFeature':
            features[field_key] = fnc(shape, dtype, allow_missing=True)
        else:
            features[field_key] = fnc(shape, dtype)

    # ----------- read dataset with built feature_dict -----------------
    dataset = tf.data.TFRecordDataset(file_prefix + '.tfrecord')

    def _parse_tf_record(example_proto):
        single_dict = tf.parse_single_example(example_proto, features)

        # restore complex dtype if it was lost upon saving
        for key in single_dict.keys():
            if feature_dicts[key]['previous_dtype'] == 'complex64':
                temp = tf.reshape(single_dict[key], shape=[-1, 2])
                single_dict[key] = tf.bitcast(temp, type=tf.complex64)
        return single_dict

    return dataset.map(_parse_tf_record)


def read_dataset(src_dir, parallel_files=1):
    # ------------find json file in directory---------------------------

    src_dir = os.path.abspath(src_dir) + '/'
    assert os.path.exists(src_dir + JSON_FNAME)


    # ---------build feature dictionary from json-----------------------
    READ_FNCS = {
        'FixedLen': tf.FixedLenFeature,
        'FixedLenSequence': tf.FixedLenSequenceFeature
    }
    STR_TO_DTYPE = {
        'tf.float32': tf.float32,
        'tf.int64': tf.int64,
        'tf.string': tf.string
    }

    with open(src_dir + JSON_FNAME) as json_file:
        feature_dicts = json.load(json_file)


    features = {}
    for field_key in feature_dicts.keys():

        # remove odd 'u' from keys as a result of json
        field_key = str(field_key)
        for sub_key in feature_dicts[field_key].keys():
            sub_key = str(sub_key)

        fnc_str = feature_dicts[field_key]['fnc']
        dtype_str = feature_dicts[field_key]['dtype']
        shape = feature_dicts[field_key]['shape']

        fnc = READ_FNCS[fnc_str]
        dtype = STR_TO_DTYPE[dtype_str]
        if fnc.__name__ == 'FixedLenSequenceFeature':
            features[field_key] = fnc(shape, dtype, allow_missing=True)
        else:
            features[field_key] = fnc(shape, dtype)


    dataset = tf.data.Dataset.list_files(src_dir + "*.tfrecord")
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=parallel_files)
    def _parse_tf_record(example_proto):
        single_dict = tf.parse_single_example(example_proto, features)

        # restore complex dtype if it was lost upon saving
        for key in single_dict.keys():
            if feature_dicts[key]['previous_dtype'] == 'complex64':
                temp = tf.reshape(single_dict[key], shape=[-1, 2])
                single_dict[key] = tf.bitcast(temp, type=tf.complex64)
        return single_dict

    return dataset.map(_parse_tf_record)



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _complex_feature(value):
    """Returns a float list of length len(value)*2 from a complex array"""
    value = value.astype(np.complex64)
    temp = value.view(np.float32)

    return tf.train.Feature(float_list=tf.train.FloatList(value=temp))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


FEAT_FNC_SWITCH = {
    "string":  _bytes_feature,
    "byte":    _bytes_feature,
    "object":  _bytes_feature,

    "float32": _float_feature,
    "float64": _float_feature,
    "complex64":  _complex_feature,
    "complex128": _complex_feature,

    "bool":   _int64_feature,
    "enum":   _int64_feature,
    "int32":  _int64_feature,
    "uint32": _int64_feature,
    "int64":  _int64_feature,
    "uint64": _int64_feature
}


def _build_features(feature_dict, keys):
    features = {}
    for i in range(len(feature_dict)):
        dtype = feature_dict[keys[i]].dtype.name
        features[keys[i]] = FEAT_FNC_SWITCH[dtype](
            feature_dict[keys[i]].ravel())
    return features


def _write_to_tf_record(batch, writer):
    NUM_SAMPS = batch.itervalues().next().shape[0]
    keys = batch.keys()
    for i in range(NUM_SAMPS):
        observation = {k: v[i] for k, v in batch.iteritems()}
        features = _build_features(observation, keys)
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        s = example.SerializeToString()
        writer.write(s)


def _build_json_dict(batch):
    assert isinstance(batch, dict)

    batch_keys = batch.keys()

    tensor_metadata = ['fnc', 'dtype', 'shape', 'previous_dtype']

    features = dict.fromkeys(batch_keys)

    for field in batch_keys:
        features[field] = dict.fromkeys(tensor_metadata)

        # get fnc to use while reading

        # excluding batch dim
        if len(batch[field].shape[1:]) == 1 and batch[field].shape[1] == 1:
            features[field]['fnc'] = 'FixedLen'
        else:
            features[field]['fnc'] = 'FixedLenSequence'

        # get shape
        features[field]['shape'] = batch[field].shape[1:]  # exclude batch dim

        # get previous dtype
        features[field]['previous_dtype'] = batch[field].dtype.name
        if 'complex' in features[field]['previous_dtype']:
            # complex128 will get casted to complex64 later
            features[field]['previous_dtype'] = 'complex64'

        # get dtype
        fnc = FEAT_FNC_SWITCH[batch[field].dtype.name]
        if 'float' in fnc.__name__:
            dtype = 'tf.float32'
        elif 'complex' in fnc.__name__:
            dtype = 'tf.float32'
        elif 'int' in fnc.__name__:
            dtype = 'tf.int64'
        elif 'bytes' in fnc.__name__:
            dtype = 'tf.string'
        else:
            raise ValueError('Cant find datatype')
        features[field]['dtype'] = dtype

    return features


def _is_normal_batch(current_batch, json_dict):
    assert isinstance(current_batch, dict)

    cb_keys = current_batch.keys()
    jd_keys = json_dict.keys()
    if set(cb_keys) != set(jd_keys):
        raise ValueError("Batch keys do not match json dictionary")
    for key in cb_keys:
        if current_batch[key].dtype.name != json_dict[key]['previous_dtype']:
            raise ValueError(
                "Batch feature datatypes do not match json dictionary")
        # exclude batch dim
        if list(current_batch[key].shape[1:]) != list(json_dict[key]['shape']):
            raise ValueError(
                "Batch feature shapes do not match json dictionary")

    return True
