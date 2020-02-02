import os
import itertools
import json
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf

import string
import random

from dataset_utils import FEAT_FNC_SWITCH, batch_to_nparray

JSON_FNAME = "TF_RECORD_FEATURES_SPEC.json"



def record_generator(batch_gen,
                     logger,
                     output_dir,
                     num_batches,
                     batches_per_file = 25,
                     unbatch = True,
                     test = False):
  """
  Parameters
  batch_generator: instance of a batch generator, must return a flat dict
  output_dir: string, path to output directory
  num_batches: int, number of batches to sink
  batches_per_file: int, number of batches per tfrecord
  
  This will write a json names `FEATURE_DATA.json` to help read the tfrecords
  later using the `replay_generator` function.
  Will also write `ceil(num_batches / batches_per_file)` tfrecords
  
  Usage:
  >>> bg = batch_generator(batch_size=32)
  >>> print(type(bg.next()))
        `dict`
  >>> record_generator(bg, 'path/to/tfrecords/', batches_per_file=50)
  >>> files = os.listdir('path/to/tfrecords/')
  >>> for f in files:
  >>>   print(f)
      'FEATURE_DICT.json'
      tmpaqi23x.tfrecord
      tmpu9cb12.tfrecord
      ...
  """
  output_dir = os.path.abspath(output_dir)
  
  # check if json feature dict already exists in dir
  json_path = os.path.join(output_dir, JSON_FNAME)
  
  # load existing json if exists
  if os.path.isfile(json_path):
    with open(json_path) as f:
      json_dict = json.load(f)
  else:
    batch = next(batch_generator)
    batch = batch_to_nparray(batch)
    batch_generator = itertools.chain([batch], batch_generator)
    if unbatch:
      batch = {k: v[0] for k, v in batch.items()}
    json_dict = _build_json_dict(batch, unbatch)
    with open(os.path.join(output_dir, JSON_FNAME), "w") as json_file:
      json.dump(json_dict, json_file)
    
  # e.g. if batches_per_file = 25, num_batches = 103,
  # we get [25, 25, 25, 25, 3]
  batches_per_file = itertools.chain(
    itertools.repeat(batches_per_file, num_batches/batches_per_file),
    [num_batches % batches_per_file])
    
  def validate_shape(batch):
    if unbatch:
      batch_sizes = [len(x) for x in batch.values()]
      assert len(set(batch_sizes)) == 1, "Not all batch dims match"
      record = {k: v[0] for k, v in batch.items()}
      _assert_item_has_expected_shpe(record, json_dict, unbatch)
    else:
      _assert_item_has_expected_shpe(record, json_dict, unbatch)
      

  def unbatch_elements(batch):
    """ Returns a numpy array o unbatched records as dicts"""
    return np.asarray([{k: v[i] for k,v in batch.items()}
                                for i in xrange(batch_size)])
  
  examples = list()
  file_no = 0
  for n in batches_per_file:
    file_no += 1
    output_record = NamedTemporaryFile(dir=output_dir,
                                       suffix='.tfrecord',
                                       delete=False)
    with tf.python_io.TFRecordWriter(output_record.name) as writer:
      for i, batch in enumerate(batch_generator):
        logger.info(
          'Processing batch: %s of %s in file %s', i, n, file_no)
        if i == n: break # Reached maximum batches_per_file
          
        # coerce records to numpy arrays
        batch = batch_to_nparray(batch)
        validate_shape(batch)
        
        batch_size = batch.values().next().shape[0]
        
        if test:
          examples.append(record)
        
        if unbatch:
          batch = unbatch_elements(batch)
          for j in xrange(batch_size):
            _assert_item_has_expected_shape(record,
                                            json_dict,
                                            unbatch)
            _write_to_tfrecord(record, writer, batch_size)
            
        else:
          _write_to_tfrecord(batch, writer, batch_size)
          
    return np.asarray(examples) if test else None
    
    
def replay_generator(src_dir, parallel_files=1):
  """
  Parameters
  src_dir: Directory containing tfrecords to be read into memory
  parallel_files: Number of files to read concurrently
  
  Reads in tensorflow records and parses examples
  Usage:
  >>> dataset = replay_generator('/my/tfrecords/path', 1)
  >>> sess = tf.Session()
  >>> it = dataset.make_one_shot_iterator()
  >>> nb = it.get_next()
  >>> type(sess.run(nb))
      <type 'dict'>
  """
  # find json file in directory
  src_dir =os.path.abspath(src_dir)
  json_path = os.path.join(src_dir, JSON_FNAME)
  assert os.path.isfile(json_path)
  
  # bild feature dictionary from json
  READ_FNCS = {
    'FixedLen': tf.FixedLenFeature,
    'FixedLenSequence': tf.FixedLenSequenceFeature
  }
  
  STR_TO_DTYPE = {
    'tf.float32': tf.float32,
    'tf.int32': tf.int32,
    'tf.int64': tf.int64,
    'tf.string': tf.string
  }

  with open(json_path, "r") as json_file:
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
    
    if fnc.__name__ = 'FixedLenSequenceFeature':
      features[field_key] = fnc(shape, dtype, allow_missing=True)
    else:
      features[field_key] = fnc(shape, dtype)
      
  dataset = tf.data.Dataset.list_files(src_dir + "/*.tfrecord")
  dataset = dataset.interleave(tf.data.TFRecordDataset,
  cycle_length=parallel_files)
  
  def _parse_tf_record(example_proto):
    single_dict = tf.parse_single_example(example_proto, features)
    # restore complex dtype if it was lost upon saving
    for key in single_dict.keys():
      if feature_dicts[key]['previous_dtype'] == 'complex64':
        temp = tf.reshape(single_dict[key], shape=[-1, 2])
        single_dict[key] = tf.bitcast(temp, type=tf.complex64) # should be dtype?
    return single_dict
  
  return dataset.map(_parse_tf_record)




def _build_features(feature_dict, keys):
  features = {}
  for i in range(len(feature_dict)):
    try:
      dtype = feature_dict[keys[i]].dtype.name
    except:
      AttributeError:
        feature_dict[keys[i]] = np.asarray(feature_dict[keys[i]])
        dtype = feature_dict[keys[i]].dtype.name
      features[keys[i]] = FEAT_FNC_SWITCH[dtype](
        feature_dict[keys[i]].ravl())
  return features
  


def _build_json_dict(batch, unbatch):
  assert isinstance(batch, dict)
  
  def _assert_is_flat_dict(_dict):
    for v in _dict.items():
      if type(v) == dict:
        raise TypeError("Batch is a nested dictionary")
      True
      
  _assert_is_flat_dict(batch)
  batch_keys = batch.keys()
  
  tensor_metadata = ['fnc', 'dtype', 'shape', 'previous_dtype']
  features = dict.fromkeys(batch_keys)
  
  for field in batch_keys:
    features[field] = dict.fromkeys(tensor_metadata)
    
    # exclude batch_dim
    if (len(batch[field].shape[1:]) == 1 and batch[field].shape[1] == 1) \
      or batch[field].shape == () or batch[field].shape == (1,):
      features[field]['fnc'] = 'FixedLen'
    else:
      features[field]['fnc'] = 'FixedLenSequence'
  
    # get shape
    features[field]['shape'] = batch[field].shape if unbatch \
                                                  else batch[field].shape[1:]
                                                  
    # get previous dtype
    fetaures[field]['previous_dtype'] = batch[field].dtype.name
    
    if 'complex' in features[field]['previous_dtype']:
      # complex128 will get casted to complex64 later
      features[field]['previous_dtype'] = 'complex64'

    # get dtype
    dtype_name = batch[field].dtype.name
    fnc = FEAT_FNC_SWITCH[dtype_name]
    if 'float' in fnc.__name__:
      dtype = 'tf.float32'
    elif 'complex' in fnc.__name__:
      dtype = 'tf.complex64'
    elif 'int' in fnc.__name__:
      dtype = 'tf.int64'
    elif 'bytes' in fnc.__name__:
      dtype = 'tf.string'
    else:
      raise ValueError("Can't find datatype")
      
    features[field]['dtype'] = dtype
    
    return features


def _assert_batch_has_expected_shape(batch, expected_shape, unbatch):
  assert isinstance(batch, dict)
  
  b_keys = batch.keys()
  e_keys = expected_shape.keys()
  if set(b_keys) != set(e_keys):
    raise ValueError("Batch keys do not match json dictionary")
  for key in b_keys:
    if type(batch[key]) is not np.ndarray and type(batch[key]) is not np.array:
      raise TypeError(
        "Batch is a singular item, not np array ... maybe you meant to set unbatch=False?")
    if batch[key].dtype.name != expected_shape[key]['previous_dtype']:
      raise ValueError(
        "Batch feature datatypes do not match json dictionary")
        
    # exclude batch dim
    if list(batch[key].shape[1:]) != list(expected_shape[key]['shape']):
      raise ValueError(
        "Batch feature shapes do not match json dictionary")

  return True



def _assert_item_has_expected_shape(item, expected_shape, unbatch):
  assert isinstance(item, dict)
  
  r_keys = item.keys()
  e_keys = expected_shape.keys()
  if set(r_keys) != set(e_keys):
    raise ValueError("Item key does not batch json dictionary")
  for key in r_keys:
    if item[key].dtype.name != expected_shape[key]['previous_dtype']:
      raise ValueError(
        "Item feature datatype does not batch json dictionary")
    item_shape = item[key].shape if unbatch else item[key].shape[1:]
    if list(item_shape) != list(expected_shape[key]['shape']):
      raise ValueError(
        "Item shape does not match json dictionary")
        
  return True



def test_generator(n=32, batch_sz=10, seq_len=8):
  pass
