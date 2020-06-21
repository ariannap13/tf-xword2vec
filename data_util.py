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
  
  def path_file(self, fname=None, subfolder=None, full_path=None):
    if (fname is None):
      raise Exception("file name is empty")
    if (full_path is None):
      full_path = curr_path
    if (subfolder is None):
      return os.path.join(full_path, fname)
    else:
      return os.path.join(full_path, subfolder, fname)
  
  def remove_last_empty(self, full_path):
    try:
      with open(full_path, encoding='utf-8') as f_input:
        data = f_input.read().rstrip('\n')
      with open(full_path, 'w', encoding='utf-8') as f_output:    
        f_output.write(data)
      except: 
        return False
    return True
  
  def save_embed_proj(self, array_embed, list_vocab, path_embed=None,
                      fname='embeddings'):
    if path_embed is None:
      path_embed = self.out_path
      
    # create dataframe
    df_embed = pd.DataFrame(array_embed, index=list_vocab)
    
    # save it in HDF5
    store = pd.HDFStore(self.path_file(fname + '.h5',, path_embed))
    store['df_embed'] = df_embed
    store.close()
    
    # save vector to Google project, for instance
    file_vec = self.path_file((fname + '.vec',, path_embed)
    df_embed.to_csv(file_vec, sep='\t', header=False, index=False)
    # remove last blank line in vector file
    self.remove_last_empty(file_vec)
    
    # create label file
    file_vec = self.path_file('project_embed.lbl',, path_embed)
    with open(file_vec, 'w', encoding="utf-8") as fw:
      for i, w in enumerate(list_vocab):
        if i > 0: fw.write('\n' + w)
        else: fw.write(w)
      fw.close()
