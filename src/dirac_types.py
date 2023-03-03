"""
Author      : Pablo Luesia-Lahoz
Date        : November 2022
File        : dirac_types.py
Description : Definition and implementation of Dirac notation datatypes
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def dagger(vector_states):
    return np.conjugate(vector_states.swapaxes(-1,-2))


def fftn_but_first(data, o_shape):
    """
    It return the Fourier transform along all the axes of the data but the
    first one, and it returns twice -1 the given shape
    @param data             : Data to transform
    @param o_shape          : Output shape of the data for axes from 1 to n.
                              If larger than the input, it pads with zeros, if 
                              lower, it trim the input
    @return (f_data, axes)  : f_data is the data in the fourier domain but the
                              first axes, and axes are the axes to transform
    """
    assert o_shape.shape[0] == data.ndim-1,\
        "The output shape does not mach with the dimensions of the data"
    axes = np.arange(1, data.ndim)
    f_data = np.fft.fftn(data, s=o_shape, axes=axes)
    return (f_data, axes)


class Ket(np.ndarray):
    """
    The representation of a quantum system |a>
    """

    def __new__(cls, *args, ket_format = False, label: str = None):
        """
        Constructor of a ket.
        @param args         : Bra, a list of elements or a numpy.ndarray.
                              If a numpy.ndarray, it will move axes to set it
                              in the ket format
        @param ket_format   : If true, it assumes the data is already in the
                              ket format
        @param label        : Label for printing the ket
        """
        if len(args) == 1 and isinstance(args[0], Bra):
            obj = dagger(args[0]).view(Ket)
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            obj = args[0].view(cls)
            if not ket_format:
                obj = np.moveaxis(obj, 0, -1)[..., np.newaxis]
        else:
            obj = np.moveaxis(np.asarray([*args]).view(cls), 
                              0, -1)[..., np.newaxis]

        obj.label = label
        return obj

    def __array_finalize__(self, obj: None | NDArray[Any], /) -> None:
        """
        Numpy array finalize
        """
        if obj is None:
            return
        elif isinstance(obj, Bra):
            self.label = None
        else:
            self.label = getattr(obj, 'label', None)

    def bra(self):
        """
        Transform the ket into a bra
        """
        return Bra(self)

    def outer(self, bra):
        """
        Return the outer product between a ket and a bra (ketbra)
        """
        assert isinstance(bra, Bra), "The second term is not a Bra"
        return np.matmul(self, bra).view(np.ndarray)

    def __str__(self):
        """
        To string overload
        """
        if self.label is None:
            prefix = ''
        else:
            prefix = '|' + self.label + '> = '
        return prefix + str(self.view(np.ndarray))

    def __mul__(self, term):
        """
        Product override
        """
        if isinstance(term, Bra):
            # Outer product
            return self.outer(term)
        else:
            return np.matmul(self.view(np.ndarray), term)


class Bra(np.ndarray):
    """
    The dual representation of a ket <a|
    """

    def __new__(cls, *args, bra_format = False, label: str = None):
        """
        Constructor of a ket.
        @param args         : Ket, a list of elements or a numpy.ndarray.
                              If a numpy.ndarray, it will move axes to set it
                              in the bra format
        @param bra_format   : If true, it assumes the data is already in the
                              bra format
        @param label        : Label for printing the ket
        """
        if len(args) == 1 and isinstance(args[0], Ket):
            obj = dagger(args[0]).view(Bra)
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            obj = args[0].view(cls)
            if not bra_format:
                obj = np.moveaxis(obj, 0, -1)[..., np.newaxis, :]
        else:
            obj = np.moveaxis(np.asarray([*args]).view(cls), 
                                0, -1)[..., np.newaxis, :]

        obj.label = label
        return obj

    def __array_finalize__(self, obj: None | NDArray[Any], /) -> None:
        """
        Numpy array finalize
        """
        if obj is None:
            return
        elif isinstance(obj, Ket):
            self.label = None
        else:
            self.label = getattr(obj, 'label', None)

    def ket(self):
        """
        Transform the bra into a ket
        """
        return Ket(self)

    def inner(self, ket):
        """
        Return the outter product between a bra and a ket (braket)
        """
        # Inner product with convolutions in Fourier
        return np.matmul(self, ket).view(np.ndarray)

    def __str__(self):
        """
        String overload
        """
        if self.label is None:
            prefix = ''
        else:
            prefix = '<' + self.label + '| = '
        return prefix + str(self.view(np.ndarray))

    def __mul__(self, term):
        """
        Multiply operator override
        """
        if isinstance(term, Ket):
            return self.inner(term)
        else:
            return np.matmul(self.view(np.ndarray), term) 
