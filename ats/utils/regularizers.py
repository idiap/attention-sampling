#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Define regularizers that might be useful for training an ATS model."""

from keras import backend as K
from keras.regularizers import Regularizer

from . import to_float32


class MultinomialEntropy(Regularizer):
    """Increase or decrease the entropy of a multinomial distribution.
    
    Arguments
    ---------
        strength: A float that defines the strength and direction of the
                  regularizer. A positive number increases the entropy, a
                  negative number decreases the entropy.
        eps: A small float to avoid numerical errors when computing the entropy
    """
    def __init__(self, strength=1, eps=1e-6):
        self.strength = to_float32(strength)
        self.eps = to_float32(eps)

    def __call__(self, x):
        logx = K.log(x+self.eps)
        return self.strength * K.sum(x * logx) / to_float32(K.shape(x)[0])

    def get_config(self):
        return {"strength": self.strength, "eps": self.eps}


def multinomial_entropy(strength=1, eps=1e-6):
    return MultinomialEntropy(strength, eps)
