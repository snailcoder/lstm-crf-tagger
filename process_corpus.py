#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : process_corpus.py
# Author            : Yan <yanwong@126.com>
# Date              : 31.03.2020
# Last Modified Date: 09.04.2020
# Last Modified By  : Yan <yanwong@126.com>

import re
import argparse

BASIC_LATIN_PROG = re.compile(r'[\u0021-\u007e]')

"""Legal chars include some of general punctuations, letterlike symbols,
CJK symbols and punctuation, CJK unified ideographs, halfwidth and
fullwidth forms.
"""
ILLEGAL_CHAR_PROG = re.compile(
    r'[^\u2014\u2018\u2019\u201c\u201d\u2026\u2030\u2103\u3001\u3002\u3007-\u300f\u4e00-\u9fff\uff01-\uff5e]')

def process_PFR_sentence(s):
  s = re.sub(r'\[|\][a-z]+', '', s)  # remove square brackets for proper nouns

  toks = s.split()
  processed = []
  for t in toks:
    parts = t.split('/')
    assert(len(parts) == 2)

    chs, pos = parts
    chs = BASIC_LATIN_PROG.sub(lambda x: chr(ord(x.group(0)) + 0xfee0), chs)
    chs = ILLEGAL_CHAR_PROG.sub('', chs)  # remove illegal chars
    if not chs:
      continue

    if len(chs) == 1:  # single char word
      processed.append([chs, 'S', pos])
    else:
      processed.append([chs[0], 'B', pos])  # begining
      for i in range(1, len(chs) - 1):
        processed.append([chs[i], 'M', pos])  # media
      processed.append([chs[-1], 'E', pos])  # end

  return processed

def extract_PFR_lines(input_file):
  # Each line is treated as a single sentence.

  result = []
  with open(input_file, 'r') as f:
    for line in f:
      line = line[23:]  # skip the number sequence at the begining
      result.append(line)

  return result

def extract_PFR_sentences(input_file):
  # Split each line is broken into some sentenecs by break punctuations.

  break_punc_prog = re.compile(r'[，、。！？：；]/w')
  inner_prog = re.compile(r'（/w([^）]+)')
  right_parenthesis_prog = re.compile(r'）/w')
  result = []

  with open(input_file, 'r') as f:
    for line in f:
      line = line[23:]  # skip the number sequence at the begining
      sents = break_punc_prog.split(line)
      to_process = []
      for s in sents:
        inner = inner_prog.findall(s)  # extract contents in parenthesis
        s = inner_prog.sub('', s)  # remove left parenthesis and contents
        s = right_parenthesis_prog.sub('', s)  # remove right parenthesis
        result.append(s)
        result.extend(inner)

  return result

def process_PFR_corpus(input_file, output_tokens,
                       output_chars=None, split_line=False):
  """
    output_chars
    save splitted chars for training word vector
  """

  sents = extract_PFR_sentences(input_file) if split_line else extract_PFR_lines(input_file)

  processed = [process_PFR_sentence(s) for s in sents]
  with open(output_tokens, 'w') as f:
    for s in processed:
      for c in s:
        f.write('%s\t%s\t%s\n' % (c[0], c[1], c[2]))
      f.write('\n')

  if output_chars is not None:
    with open(output_chars, 'w') as f:
      for s in processed:
        for c in s:
          f.write(c[0] + ' ')
        f.write('\n')

def process_SIGHAN2005_corpus(input_file, output_tokens, output_chars=None):
  """
    output_chars
    save splitted chars for training word vector
  """

  processed = []

  with open(input_file, 'r') as f:
    for s in f:
      toks = s.split()
      chs = []
      for t in toks:
        t = BASIC_LATIN_PROG.sub(lambda x: chr(ord(x.group(0)) + 0xfee0), t)
        t = ILLEGAL_CHAR_PROG.sub('', t)  # remove illegal chars

        if not t:
          continue
        if len(t) == 1:
          chs.append([t, 'S'])
        else:
          chs.append([t[0], 'B'])
          for i in range(1, len(t) - 1):
            chs.append([t[i], 'M'])
          chs.append([t[-1], 'E'])

      processed.append(chs)

  with open(output_tokens, 'w') as f:
    for s in processed:
      for c in s:
        f.write('%s\t%s\n' % (c[0], c[1]))
      f.write('\n')

  if output_chars is not None:
    with open(output_chars, 'w') as f:
      for s in processed:
        for c in s:
          f.write(c[0] + ' ')
        f.write('\n')

def main():
  parser = argparse.ArgumentParser(
      description='Process the corpus: PFR or SIGHAN2005 bake-off. '
           'The original corpus should be encoded as UTF8.')
  parser.add_argument('input', help='The corpus file.')
  parser.add_argument('type', help='The corpus type: PFR or SIGHAN2005.')
  parser.add_argument('output',
                      help='The output file containing characters and tagss.')
  parser.add_argument('-chars',
                      help='Save the original corpus in a '
                      'word2vec-training-ready format: characters are '
                      'separated by a space.')
  args = parser.parse_args()

  if args.type == 'PFR':
    process_PFR_corpus(args.input, args.output, args.chars)
  elif args.type == 'SIGHAN2005':
    process_SIGHAN2005_corpus(args.input, args.output, args.chars)
  else:
    raise ValueError('The corput type should be one of these types: '
                     'PFR, SIGHAN2005.')

if __name__ == '__main__':
  main()

