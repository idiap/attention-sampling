#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""The data module provides interfaces and implementations to generate tf
tensors for patches of multi resolution data.

We have the following assumptions:

1. The data have n spatial dimensions (e.g. 1 for sound, 2 for images, etc.)
2. The data can be expressed in a number of different discrete scales which we
   call **levels**.
3. Each level depicts the same spatial region which means that we have a
   trivial correspondence between levels by associating each value with (1/s)^n
   values of the higher resolution centered around that value. n is the number
   of dimensions, s is the scale down.
"""

from .from_tensors import FromTensors
