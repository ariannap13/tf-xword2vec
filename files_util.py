# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:47:23 2020
File Tools
@author: eduar
"""
import pandas as pd
import os

class DataFileTools(object):
  """ Data file tools.
  """
  
  curr_path = os.path.dirname(os.path.abspath(__file__))
  
  def __init__(self, in_path=curr_path, out_path=curr_path, log_path=curr_path,
               log_ext=".log", vec_ext=".vec", tag_ext=".lbl"):
    """Constructor.
    Args:
      in_path: string scalar, default path to read files.
      out_path: string scalar, default path to save files.
      log_ext: string scalar, default log extension.
    """
    self._in_path = in_path
    self._out_path = out_path
    self._log_path = log_path
    self._log_ext = log_ext
    self._vec_ext = vec_ext
    self._tag_ext = tag_ext
    
  @property
  def log_ext(self):
    return self._log_ext
  
  
  def time_sufix(self, extension):
      from time import strftime
      sufix = "_" + strftime("%Y-%m-%d_%H-%M")
      sufix = sufix + extension
      return sufix
  
  def remove_last_empty(self, full_path):
    try:
      with open(full_path, encoding='utf-8') as f_input:
        data = f_input.read().rstrip('\n')
      with open(full_path, 'w', encoding='utf-8') as f_output:    
        f_output.write(data)
      except: return False
      return True
  
  def save_embed_proj(self, array_embed, list_vocab, path_embed,
                      fname='embeddings'):
    # create dataframe
    df_embed = pd.DataFrame(array_embed, index=list_vocab)
    # save it in HDF5
    store = pd.HDFStore(os.path.join(path_embed, fname + '.h5'))
    store['df_embed'] = df_embed
    store.close()
    # save vector to Google project, for instance
    df_embed.to_csv(os.path.join(path_embed, fname + '.vec'),
                    sep='\t', header=False, index=False)
    
    # remove last blank line in vector file
    full_path = os.path.join(path_embed, fname + '.vec')
    rep = self.remove_last_empty(full_path)
    
    # create label file
    os.path.join(path_embed, 'project_embed.lbl')
    with open(full_path, 'w', encoding="utf-8") as fw:
      for i, w in enumerate(list_vocab):
        if i > 0: fw.write('\n' + w)
        else: fw.write(w)
      fw.close()
  