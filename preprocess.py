#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : preprocess.py
# Author            : Yan <yanwong@126.com>
# Date              : 31.03.2020
# Last Modified Date: 03.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import re
import collections

def process_PFR_sentence(s):
  s = re.sub(
      r'([\uff10-\uff19]+\uff0f[\uff10-\uff19]+)|([\uff10-\uff19]+(\uff0e[\uff10-\uff19]+)?\uff05?)',
      r'N', s)  # replace fullwidth digits
  s = re.sub(
      r'[\uff21-\uff3a,\uff41-\uff5a]+',
      r'L', s)  # replace fullwidth latins
  s = re.sub(r'\[|\][a-z]+', '', s)  # remove square brackets for proper nouns
  toks = s.split()
  special = {'N': '<NUM>', 'L': '<LAT>'}
  processed = []
  punc_prog = re.compile(r'.+/w')
  for t in toks:
    if punc_prog.match(t):
      continue
    parts = t.split('/')
    assert(len(parts) == 2)
    chs, pos = parts
    assert(len(chs) > 0 and len(pos) > 0)
    if len(chs) == 1:  # single char word
      processed.append([chs, 'S', pos])
    else:
      processed.append([chs[0], 'B', pos])  # begining
      for i in range(1, len(chs) - 1):
        processed.append([chs[i], 'M', pos])  # media
      processed.append([chs[-1], 'E', pos])  # end

  for p in processed:
    p[0] = special.get(p[0], p[0])  # replace with special tags

  return processed

def process_PFR_corpus(input_file, output_file):
  break_punc_prog = re.compile(r'[，、。！？：；]/w')
  inner_prog = re.compile(r'（/w([^）]+)')
  right_parenthesis_prog = re.compile(r'）/w')
  processed = []

  with open(input_file, 'r') as f:
    for line in f:
      line = line[23:]  # skip the number sequence at the begining
      sents = break_punc_prog.split(line)
      to_process = []
      for s in sents:
          inner = inner_prog.findall(s)  # extract contents in parenthesis
          s = inner_prog.sub('', s)  # remove left parenthesis and contents
          s = right_parenthesis_prog.sub('', s)  # remove right parenthesis
          to_process.append(s)
          to_process.extend(inner)
      for s in to_process:
        processed.append(process_PFR_sentence(s))

  with open(output_file, 'w') as f:
    for s in processed:
      for t in s:
        f.write('%s\t%s\t%s\n' % (t[0], t[1], t[2]))
      f.write('\n')

# def _build_vocab(input_file):
#   word_cnt = collections.Counter()
# 
#   with tf.io.gfile.GFile(input_file, mode='r') as f:
#     for line in f:
#       word_cnt.update(line.split())

process_PFR_corpus('testdata.txt', 'processed.txt')

