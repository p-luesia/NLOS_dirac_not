"""
Author      : Pablo Luesia-Lahoz
Date        : November 2022
File        : dirac_types.py
Description : Definition and implementation of Dirac notation datatypes
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any


def dagger(vector_states):
    return np.conjugate(vector_states)


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
    f_data = np.fft.fftn(data, s = o_shape, axes = axes)
    return (f_data, axes)


class Ket(np.ndarray):
    """
    The representation of a quantum system |a>
    """

    def __new__(cls, *args, label: str = None):
        """
        Constructor of a ket.
        @param args     : Bra or a list of numbers
        @param label    : Label for printing the ket
        """
        if len(args) == 1 and isinstance(args[0], Bra):
            obj = dagger(args[0]).view(Ket)
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            obj = args[0].view(cls)
        else:
            obj = np.asarray([*args]).view(cls)

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
        shape_bra = np.array(bra.shape)
        shape_ket = np.array(self.shape)
        assert np.all(shape_bra[1:] == shape_ket[1:]),\
                "Bra and Ket shapes do not match"

        # Data to Fourier domain
        o_shape = shape_bra[1:]*2 - 1
        f_ket, axes = fftn_but_first(self, o_shape)
        f_bra, _ = fftn_but_first(bra, o_shape)

        # Outter product with convolutions in Fourier
        f_o = f_ket[:, np.newaxis, ...] * f_bra[np.newaxis, ...]

        # The output expected shape
        o_shape = tuple(shape_bra[1:])
        return np.fft.ifftn(f_o, s=o_shape, axes = axes+1).view(np.ndarray)#.copy()

    def __str__(self):
        """
        To string overload
        """
        if self.label is None:
            prefix = ''
        else:
            prefix = '|' + self.label + '> = '
        return prefix + str(self.view(np.ndarray))

    def __mul__(self, bra):
        """
        Outer product override
        """
        return self.outer(bra)

    # def __rmul__(self, term):
    #     """
    #     Multiply operator override
    #     """
    #     return np.matmul(term, self)


class Bra(np.ndarray):
    """
    The dual representation of a ket <a|
    """

    def __new__(cls, *args, label: str = None):
        """
        Constructor of a ket.
        @param args     : Bra or a list of numbers
        @param label    : Label for printing the ket
        """
        if len(args) == 1 and isinstance(args[0], Ket):
            obj = dagger(args[0]).view(Bra)
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            obj = args[0].view(cls)
        else:
            obj = np.asarray([*args]).view(cls)

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
        assert isinstance(ket, Ket), "The second term is not a Bra"
        assert ket.shape == self.shape, "Bra and Ket shapes do not match"

        # Data to Fourier domain
        f_bra, axes = fftn_but_first(self)
        f_ket, _ = fftn_but_first(ket)

        # Inner product with convolutions in Fourier
        f_o = np.sum(f_ket * f_bra, axis = 0)

        # The output expected shape
        return np.fft.ifftn(f_o)

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
            return np.inner(self, term)

