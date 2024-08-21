#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys

import sympy
import numpy as np

from typing import List
from utils.log_config import logger


class SymbolicLGPDS:
    """ Build a lyapunov function using the sum of squares technique. The class
    is ultimately employed to verify the global or local stability of a polynomial
    dynamical system around the target.

    """

    def __init__(self, degree: int = 4, dim: int = 2, convex_lpf: bool = False):
        """ Initialize an SOSStability class object.

        Args:
            degree (int, optional): Complexity of the LPF polynomial. Defaults to 4.
            dim (int, optional): Dimension of the problem. Defaults to 2.
            convex_lpf (bool, optional): Come up with a convex LPF to avoid spurious attractors.
        """

        # state variables
        self.__dim = dim
        self.__x = sympy.symbols(f'x:{self.__dim}')

        # lpf parameters
        self.__p: sympy.NumberSymbol = sympy.symbols(f'p:{self.__dim}')
        self.__q: sympy.MatrixSymbol = sympy.MatrixSymbol('q',
                                                          dim * (degree // 2),
                                                          dim * (degree // 2))

        self.__basis_vec_lpf: sympy.Matrix = self._generate_lpf_basis(degree)

        # symbolic lpf and ds definition (order matters!)
        self.__lpf = self._build_symbolic_lpf(convex_lpf)
        self.__ds = self._define_symbolic_ds()
        self.__dlpf_dt = self._build_dlpf_dt()

        # create functions
        self.__ds_func = self._ds_lambdify()
        self.__lpf_func = self._lpf_lambdify()
        self.__dlpf_dt_func = self._dlpf_dt_lambdify()

    def ds(self, x: np.ndarray, p: np.ndarray, q: np.ndarray):
        """ Getter for ds formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  dimension).
        """

        return np.array([self.__ds_func[idx](x, p[idx], q)
                         for idx in range(self.__dim)]).reshape(x.shape)

    def lpf(self, x: np.ndarray, q: np.ndarray):
        """ Getter for lpf formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  1).
        """

        return self.__lpf_func(x, q)

    def dlpf_dt(self, x: np.ndarray, p: np.ndarray, q: np.ndarray):
        """ Getter for lpf formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  1).
        """

        return self.__dlpf_dt_func(x, p, q)

    def _ds_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return [sympy.lambdify([self.__x, self.__p[idx], self.__q], self.__ds[idx].as_expr())
                for idx in range(self.__dim)]

    def _lpf_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return sympy.lambdify([self.__x, self.__q], self.__lpf.as_expr())

    def _dlpf_dt_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return sympy.lambdify([self.__x, self.__p, self.__q], self.__dlpf_dt[0].as_expr())

    def _define_symbolic_ds(self):
        """ Define the ds to be the gradient of LPF.

        sympy.Poly, sympy.Poly: ds_x, ds_y
        """

        # get lpf grad
        lpf_grad = self._lpf_gradient()

        # generate DS based on negative of lpf grad
        ds = [-self.__p[idx] * lpf_grad[idx] for idx in range(self.__dim)]
        logger.info(f'ds: \n {ds} \n')

        return ds

    def _generate_lpf_basis(self, degree: int):
        max_pow = degree // 2
        basis_vecs: List[sympy.symbols] = []
        for idx in range(self.__dim):
            basis_vecs += [self.__x[idx] ** pow for pow in range(1, max_pow + 1)]

        basis_vecs = sympy.Matrix(basis_vecs)
        logger.info(f'State vector lpf {basis_vecs.shape}: \n {basis_vecs}\n')
        return basis_vecs

    def _build_symbolic_lpf(self, is_convex: bool):
        """ Build a generalized symbolic LPF.

        Returns:
            sympy.Poly, sympy.Poly: subspace-specific components of LPF function
        """
        if is_convex: raise NotImplementedError(f'No support for strict convexity right now!')

        lpf = sympy.MatMul(self.__basis_vec_lpf.transpose(), self.__q, self.__basis_vec_lpf)
        lpf = sympy.Poly(lpf.as_explicit()[0], *self.__x)
        logger.info(f'lpf: \n {lpf} \n')
        return lpf

    def _build_dlpf_dt(self):
        """ Build  the dlpf/dt variable just for comparison.
        """
        lpf_grads = self._lpf_gradient()

        dlpf_dt = [sympy.Poly(lpf_grads[idx] * self.__ds[idx], self.__x)
                   for idx in range(self.__dim)]
        logger.info(f'dlpf_dt: \n{dlpf_dt}\n')
        return dlpf_dt

    def _lpf_gradient(self):
        """ Calculate the symbolic gradient for LPF.
        """
        lpf_grad = [sympy.diff(self.__lpf, self.__x[idx]) for idx in range(self.__dim)]
        logger.info(f'dlpf_dx: \n{lpf_grad}\n')
        return lpf_grad

    @property
    def n_params(self):
        """ Get the total number of parameters separately.
        """
        n_ply_params = self.__dim
        n_lpf_params = np.prod(np.array(self.sos_params["q"].shape))
        return n_ply_params, n_lpf_params

    @property
    def sos_params(self):
        """ Store sympy params in a dictionary for interfacing.
        """
        sympy_params_dict = dict()
        sympy_params_dict["x"] = self.__x

        sympy_params_dict["q"] = self.__q
        sympy_params_dict["p"] = self.__p

        sympy_params_dict["lpf_basis"] = self.__basis_vec_lpf
        return sympy_params_dict


def main_test():
    # define a stability obj
    sos = SymbolicLGPDS()

    # function test for ds
    x = np.array([2, 3])
    p = np.random.rand(2, 1)
    q = np.random.rand(4, 4)
    logger.info(f'DS result: {sos.ds(x, p, q)}')
    logger.info(f'LPF result: {sos.lpf(x, q)}')


class SymbolicLGPDSWOP:
    def __init__(self, degree: int = 4, dim: int = 2, convex_lpf: bool = False):

        # state variables
        self.__dim = dim
        self.__x = sympy.symbols(f'x:{self.__dim}')

        # lpf parameters
        self.__q: sympy.MatrixSymbol = sympy.MatrixSymbol('q',
                                                          dim * (degree // 2),
                                                          dim * (degree // 2))

        self.__basis_vec_lpf: sympy.Matrix = self._generate_lpf_basis(degree)

        # symbolic lpf and ds definition (order matters!)
        self.__lpf = self._build_symbolic_lpf(convex_lpf)
        self.__ds = self._define_symbolic_ds()
        self.__dlpf_dt = self._build_dlpf_dt()

        # create functions
        self.__ds_func = self._ds_lambdify()
        self.__lpf_func = self._lpf_lambdify()
        self.__dlpf_dt_func = self._dlpf_dt_lambdify()

    def ds(self, x: np.ndarray, q: np.ndarray):
        """ Getter for ds formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  dimension).
        """

        return np.array([self.__ds_func[idx](x, q)
                         for idx in range(self.__dim)]).reshape(x.shape)

    def lpf(self, x: np.ndarray, q: np.ndarray):
        """ Getter for lpf formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  1).
        """

        return self.__lpf_func(x, q)

    def dlpf_dt(self, x: np.ndarray, q: np.ndarray):
        """ Getter for lpf formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  1).
        """

        return self.__dlpf_dt_func(x, q)

    def _ds_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return [sympy.lambdify([self.__x, self.__q], self.__ds[idx].as_expr())
                for idx in range(self.__dim)]

    def _lpf_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return sympy.lambdify([self.__x, self.__q], self.__lpf.as_expr())

    def _dlpf_dt_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return sympy.lambdify([self.__x, self.__q], self.__dlpf_dt[0].as_expr())

    def _define_symbolic_ds(self):
        """ Define the ds to be the gradient of LPF.

        sympy.Poly, sympy.Poly: ds_x, ds_y
        """

        # get lpf grad
        lpf_grad = self._lpf_gradient()

        # generate DS based on negative of lpf grad
        ds = [-1 * lpf_grad[idx] for idx in range(self.__dim)]
        logger.info(f'ds: \n {ds} \n')

        return ds

    def _generate_lpf_basis(self, degree: int):
        max_pow = degree // 2
        basis_vecs: List[sympy.symbols] = []
        for idx in range(self.__dim):
            basis_vecs += [self.__x[idx] ** pow for pow in range(1, max_pow + 1)]

        basis_vecs = sympy.Matrix(basis_vecs)
        logger.info(f'State vector lpf {basis_vecs.shape}: \n {basis_vecs}\n')
        return basis_vecs

    def _build_symbolic_lpf(self, is_convex: bool):
        """ Build a generalized symbolic LPF.

        Returns:
            sympy.Poly, sympy.Poly: subspace-specific components of LPF function
        """
        if is_convex: raise NotImplementedError(f'No support for strict convexity right now!')

        lpf = sympy.MatMul(self.__basis_vec_lpf.transpose(), self.__q, self.__basis_vec_lpf)
        lpf = sympy.Poly(lpf.as_explicit()[0], *self.__x)
        logger.info(f'lpf: \n {lpf} \n')
        return lpf

    def _build_dlpf_dt(self):
        """ Build  the dlpf/dt variable just for comparison.
        """
        lpf_grads = self._lpf_gradient()

        dlpf_dt = [sympy.Poly(lpf_grads[idx] * self.__ds[idx], self.__x)
                   for idx in range(self.__dim)]
        logger.info(f'dlpf_dt: \n{dlpf_dt}\n')
        return dlpf_dt

    def _lpf_gradient(self):
        """ Calculate the symbolic gradient for LPF.
        """
        lpf_grad = [sympy.diff(self.__lpf, self.__x[idx]) for idx in range(self.__dim)]
        logger.info(f'dlpf_dx: \n{lpf_grad}\n')
        return lpf_grad

    def _lpf_hessian(self):
        """ Calculate the symbolic gradient for LPF.
        """
        lpf_hess = sympy.hessian(self.__lpf, self.__x)
        logger.info(f'd2lpf_dx: \n{lpf_hess}\n')
        return lpf_hess

    @property
    def n_params(self):
        """ Get the total number of parameters separately.
        """
        n_ply_params = 0
        n_lpf_params = np.prod(np.array(self.sos_params["q"].shape))
        return n_ply_params, n_lpf_params

    @property
    def sos_params(self):
        """ Store sympy params in a dictionary for interfacing.
        """
        sympy_params_dict = dict()
        sympy_params_dict["x"] = self.__x
        sympy_params_dict["q"] = self.__q
        sympy_params_dict["lpf_basis"] = self.__basis_vec_lpf
        return sympy_params_dict


# main entry
if __name__ == "__main__":
    main_test()