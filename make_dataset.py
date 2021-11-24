# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pickle
import re
from urllib.parse import unquote
from tqdm import tqdm


DATASET = 'yfcc100m_dataset.txt'

cleanhtml = re.compile('<a.*?>|</a>|<b>|</b>|<i>|</i>')
cleanurl = re.compile('http\S+|www\S+')

print('=> loading YFCC image ids')
image_ids = np.load('flickr_unique_ids.npy')
image_ids = set(image_ids)

print('=> loading CLIP image ids')
clip_ids = set()
with open('yfcc100m_subset_data.tsv') as f:
    for l in tqdm(f.readlines()):
        row = l.strip().split('\t')
        clip_ids.add(int(row[0]))

print('=> collecting and cleaning subset captions')
captioned = []
uncaptioned = []
with open('yfcc100m_dataset.txt') as f:
    for l in tqdm(f.readlines()):
        row = l.strip().split('\t')
        if int(row[0]) in image_ids:
            uncaptioned.append(int(row[0]))
            if int(row[0]) in clip_ids:
                title = unquote(row[8]).replace('+', ' ')
                title = re.sub(cleanhtml, '', title)
                title = re.sub(cleanurl, '', title)

                desc = unquote(row[9]).replace('+', ' ')
                desc = re.sub(cleanhtml, '', desc)
                desc = re.sub(cleanurl, '', desc)
                
                captioned.append((int(row[0]), title, desc))

with open('yfcc15m.pkl', 'wb') as f:
    pickle.dump(captioned, f)

with open('yfcc100m.pkl', 'wb') as f:
    pickle.dump(uncaptioned, f)

print('Total captioned images:', len(captioned))  # 14689580
print('Total uncaptioned images:', len(uncaptioned))  # 95920149
