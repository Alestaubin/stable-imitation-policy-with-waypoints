#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys

import sympy
import numpy as np

from typing import List
from utils.log_config import logger


class SymbolicPLYDS:
    """ Build a lyapunov function using the sum of squares technique. The class
    is ultimately employed to verify the global or local stability of a polynomial
    dynamical system around the target.

    """

    def __init__(self, ds_deg: int = 2, lpf_deg: int = 2, dim: int = 2,
                 quadratic_lpf: bool = True, simplify_lpf: bool = False):
        """ Initialize an SOSStability class object.

        TODO: Everything to 2d without loss of generality.
            Convert to 3d after successful experiments.

        Args:
            ds_deg (int, optional): Degree of the DS policy for motion generation. Defaults to 2.
            lpf_deg (int, optional): Complexity of the LPF polynomial. Defaults to 2.
            dim (int, optional): Dimension of the problem. Defaults to 2.
            quadratic_lpf (bool, optional): Come up with a quadratic LPF, still a polynomial
                but the lpf_deg is ignored. Defaults to True.
        """

        # assert the correct matrix shape
        self.__lpf_deg = lpf_deg
        self.__ds_deg = ds_deg
        self.__dim = dim
        self.__simplify_lpf = simplify_lpf

        # local vars
        self.__basis_vec_ds: sympy.Matrix = None
        self.__basis_vec_lpf: sympy.Matrix = None
        self.__px, self.__py, self.__qx, self.__qy = None, None, None, None

        # state variables
        if self.__dim == 2:
            self.__xsym, self.__ysym = sympy.symbols('x y')
        else:
            self.__xsym, self.__ysym, self.__zsym = sympy.symbols('x y z')

        # symbolic ds
        self.__dsx, self.__dsy = self._build_symbolic_ds()
        self.__dsx_func, self.__dsy_func = self._ds_lambdify()

        # symbolic lyapunov function
        if quadratic_lpf:
            logger.warning(f'Limiting stability criteria to parameterized quadratic LPF')
            self.__lpfx, self.__lpfy = self._build_quadratic_lpf()
            self.__lpfx_func, self.__lpfy_func = self._lpf_lambdify()

        else:
            self.__lpfx, self.__lpfy = self._build_symbolic_lpf()
            self.__lpfx_func, self.__lpfy_func = self._lpf_lambdify()

        # symbolic lyapunov derivatives
        self.__dlpfx_dt, self.__dlpfy_dt = self._lpf_derivation()
        self.__dlpfx_dt_func, self.__dlpfy_dt_func = self._dlpf_dt_lambdify()

    def arrange_constraints(self):
        """ Arrange lyapunov constraints based on SOS stability.
        """

        # convert dlpf_dt to ply
        dlpfx_dt_poly = sympy.Poly(self.__dlpfx_dt[0], self.__xsym, self.__ysym)
        dlpfy_dt_poly = sympy.Poly(self.__dlpfy_dt[0], self.__xsym, self.__ysym)

        # find an upper bound for the second lyapunov condition's degree
        max_pow = (sympy.total_degree(dlpfx_dt_poly) // 2) + 1

        # form a second basis for coefficient matching and finding the affine function
        basis_vec_x = [self.__xsym ** pow for pow in range(1, max_pow + 1)]
        basis_vec_y = [self.__ysym ** pow for pow in range(1, max_pow + 1)]
        self.__basis_vec_der = sympy.Matrix([1] + basis_vec_x + basis_vec_y)
        logger.info(f'State vector sos_der {self.__basis_vec_der.shape}: \n {self.__basis_vec_der}\n')

        # build coefficient matrices
        der_mat_size = len(self.__basis_vec_der)
        self.__gx: sympy.Matrix = sympy.MatrixSymbol('gx', der_mat_size, der_mat_size)
        self.__gy: sympy.Matrix = sympy.MatrixSymbol('gy', der_mat_size, der_mat_size)

        # find a basis multiplication for coefficient matching
        sos_derx = sympy.MatMul(self.__basis_vec_der.transpose(), self.__gx, self.__basis_vec_der).as_explicit()
        sos_dery = sympy.MatMul(self.__basis_vec_der.transpose(), self.__gy, self.__basis_vec_der).as_explicit()
        logger.info(f'sos_der_x ({sos_dery.shape}): \n {sympy.Poly(sos_derx[0], self.__xsym, self.__ysym)} \n')

        # convert sos derivatives to ply
        sos_derx_poly = sympy.Poly(sos_derx[0], self.__xsym, self.__ysym)
        sos_dery_poly = sympy.Poly(sos_dery[0], self.__xsym, self.__ysym)

        max_deg_derx = sympy.total_degree(sos_derx_poly)
        max_deg_dery = sympy.total_degree(sos_dery_poly)
        max_deg_der = max_deg_derx # can be changed to max of both x, y later

        # go over monomials and match coefficients
        self.__constraints_x: List = []
        self.__constraints_y: List = []

        logger.debug(f'Pringting coefficient matching sequence')
        for max_deg in range(max_deg_der + 1):
            for x_deg in range(max_deg + 1):
                y_deg = max_deg - x_deg
                monomial = self.__xsym ** x_deg * self.__ysym ** y_deg

                sos_derx_coeff = sos_derx_poly.coeff_monomial(monomial)
                dlpfx_dt_coeff = dlpfx_dt_poly.coeff_monomial(monomial)
                logger.debug(f'\ni. monomial: {monomial} \nii. sos_derx_Coeff: {sos_derx_coeff} \niii. dlpf_dt_Coeff: {dlpfx_dt_coeff}')

                sos_dery_coeff = sos_dery_poly.coeff_monomial(monomial)
                dlpfy_dt_coeff = dlpfy_dt_poly.coeff_monomial(monomial)

                if sos_derx_coeff is not sympy.S.Zero:
                    self.__constraints_x.append([dlpfx_dt_coeff, sos_derx_coeff] \
                        if dlpfx_dt_coeff is not sympy.S.Zero else [sos_derx_coeff])

                if sos_dery_coeff is not sympy.S.Zero:
                    self.__constraints_y.append([dlpfy_dt_coeff, sos_dery_coeff] \
                        if dlpfy_dt_coeff is not sympy.S.Zero else [sos_dery_coeff])

        logger.debug(f'Matching sequence concluded\n')

    def ds(self, x: np.ndarray, px: np.ndarray, py: np.ndarray):
        """ Getter for ds formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  dimension).
        """

        return np.array([self.__dsx_func(*x, px), self.__dsy_func(*x, py)]).reshape(x.shape)

    def lpf(self, x: np.ndarray, qx: np.ndarray, qy: np.ndarray):
        """ Getter for lpf formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  1).
        """

        return self.__lpfx_func(*x, qx)

    def dlpf_dt(self, x: np.ndarray, px: np.ndarray, py: np.ndarray, qx: np.ndarray,
                qy: np.ndarray):
        """ Getter for lpf formulations.

        Args:
            x (np.ndarray): Input vector that is (1,  dimension).

        Returns:
            np.ndarray: the output vector (1,  1).
        """

        return self.__dlpfx_dt_func(*x, px, py, qx)

    def _ds_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return [sympy.lambdify([self.__xsym, self.__ysym, self.__px], self.__dsx),
            sympy.lambdify([self.__xsym, self.__ysym, self.__py], self.__dsy)]

    def _lpf_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return [sympy.lambdify([self.__xsym, self.__ysym, self.__qx], self.__lpfx),
            sympy.lambdify([self.__xsym, self.__ysym, self.__qy], self.__lpfy)]

    def _dlpf_dt_lambdify(self):
        """ Lambdify the ds and return as a list.

        Returns:
            List: the output vector of lambdify functions.
        """

        return [sympy.lambdify([self.__xsym, self.__ysym, self.__px, self.__py, self.__qx],
                               self.__dlpfx_dt),
                sympy.lambdify([self.__xsym, self.__ysym, self.__px, self.__py, self.__qy],
                               self.__dlpfy_dt)]


    def _build_symbolic_ds(self):
        max_pow = self.__ds_deg // 2
        basis_vec_x = [self.__xsym ** pow for pow in range(1, max_pow + 1)]
        basis_vec_y = [self.__ysym ** pow for pow in range(1, max_pow + 1)]
        self.__basis_vec_ds = sympy.Matrix([1] + basis_vec_x + basis_vec_y)
        logger.info(f'State vector ds {self.__basis_vec_ds.shape}: \n {self.__basis_vec_ds}\n')

        ds_mat_size = len(self.__basis_vec_ds)
        self.__px: sympy.Matrix = sympy.MatrixSymbol('px', ds_mat_size, ds_mat_size)
        self.__py: sympy.Matrix = sympy.MatrixSymbol('py', ds_mat_size, ds_mat_size)

        # generate DS parameters b.T * P_i * b
        dsx = sympy.MatMul(self.__basis_vec_ds.transpose(), self.__px,
                           self.__basis_vec_ds).as_explicit()
        dsy = sympy.MatMul(self.__basis_vec_ds.transpose(), self.__py,
                           self.__basis_vec_ds).as_explicit()
        logger.info(f'ds_x ({dsy.shape}): \n {sympy.Poly(dsx[0], self.__xsym, self.__ysym)} \n')

        return dsx, dsy

    def _build_symbolic_lpf(self):
        """ Build a generalized symbolic LPF.

        Returns:
            sympy.Poly, sympy.Poly: subspace-specific components of LPF function
        """

        max_pow = self.__lpf_deg // 2
        basis_vec_x = [self.__xsym ** pow for pow in range(1, max_pow + 1)]
        basis_vec_y = [self.__ysym ** pow for pow in range(1, max_pow + 1)]
        self.__basis_vec_lpf = sympy.Matrix(basis_vec_x + basis_vec_y)
        logger.info(f'State vector lpf {self.__basis_vec_lpf.shape}: \n {self.__basis_vec_lpf}\n')

        lpf_mat_size = len(self.__basis_vec_lpf)
        self.__qx: sympy.Matrix = sympy.MatrixSymbol('qx', lpf_mat_size, lpf_mat_size)
        self.__qy: sympy.Matrix = sympy.MatrixSymbol('qy', lpf_mat_size, lpf_mat_size)

        lpfx = sympy.MatMul(self.__basis_vec_lpf.transpose(), self.__qx,
                            self.__basis_vec_lpf).as_explicit()
        lpfy = sympy.MatMul(self.__basis_vec_lpf.transpose(), self.__qy,
                            self.__basis_vec_lpf).as_explicit()
        logger.info(f'lpf_x ({lpfy.shape}): \n {sympy.Poly(lpfx[0], self.__xsym, self.__ysym)} \n')

        return lpfx, lpfy

    def _build_quadratic_lpf(self):
        """ Build a quadratic symbolic LPF.

        Returns:
            sympy.Poly, sympy.Poly: subspace-specific components of LPF function
        """

        self.__basis_vec_lpf = sympy.Matrix([self.__xsym, self.__ysym])
        logger.info(f'State vector LPF {self.__basis_vec_lpf.shape}: \n {self.__basis_vec_lpf}\n')

        lpf_mat_size = len(self.__basis_vec_lpf)

        if not self.__simplify_lpf:
            self.__qx: sympy.Matrix = sympy.MatrixSymbol('qx', lpf_mat_size, lpf_mat_size)
            self.__qy: sympy.Matrix = sympy.MatrixSymbol('qy', lpf_mat_size, lpf_mat_size)
        else:
            logger.warning(f'Further simplification of LPF in motion')
            self.__qx: sympy.Matrix = sympy.Identity(lpf_mat_size)
            self.__qy: sympy.Matrix = sympy.Identity(lpf_mat_size)

        lpfx = sympy.MatMul(self.__basis_vec_lpf.transpose(),
                            self.__qx, self.__basis_vec_lpf).as_explicit()
        lpfy = sympy.MatMul(self.__basis_vec_lpf.transpose(),
                            self.__qy, self.__basis_vec_lpf).as_explicit()
        logger.info(f'lpf_x ({lpfy.shape}): \n {sympy.Poly(lpfx[0], self.__xsym, self.__ysym)} \n')

        return lpfx, lpfy

    def _lpf_derivation(self):
        """ We do not match constraints at this point!
        """
        # lpfx partial derivatives
        dlpfx_dx = sympy.diff(self.__lpfx, self.__xsym)
        dlpfx_dy = sympy.diff(self.__lpfx, self.__ysym)
        logger.info(f'dlpf_x/dx: \n{sympy.Poly(dlpfx_dx[0], self.__xsym, self.__ysym)}\n')

        # lpfy partial derivatives
        dlpfy_dx = sympy.diff(self.__lpfy, self.__xsym)
        dlpfy_dy = sympy.diff(self.__lpfy, self.__ysym)

        # lpf derivatives
        dlpfx_dt = (sympy.MatMul(dlpfx_dx, self.__dsx) + sympy.MatMul(dlpfx_dy, self.__dsy)).as_explicit()
        dlpfy_dt = (sympy.MatMul(dlpfy_dx, self.__dsx) + sympy.MatMul(dlpfy_dy, self.__dsy)).as_explicit()
        logger.info(f'dlpf_x/dt: \n{sympy.Poly(dlpfx_dt[0], self.__xsym, self.__ysym)}\n')

        return dlpfx_dt, dlpfy_dt

    @property
    def n_params(self):
        """ Get the total number of parameters separately.
        """
        n_ply_params = self.__dim * (self.sos_params["px"].shape[0]) ** 2
        n_lpf_params = self.__dim * (self.sos_params["qx"].shape[0]) ** 2  \
            if not self.__simplify_lpf else 0
        n_dlpf_dt_params = self.__dim * (self.sos_params["gx"].shape[0]) ** 2
        return n_ply_params, n_lpf_params, n_dlpf_dt_params

    @property
    def sos_params(self):
        """ Store sympy params in a dictionary for interfacing.
        """

        sympy_params_dict = dict()
        sympy_params_dict["x"] = self.__xsym
        sympy_params_dict["y"] = self.__ysym

        sympy_params_dict["qx"] = self.__qx
        sympy_params_dict["qy"] = self.__qy

        sympy_params_dict["px"] = self.__px
        sympy_params_dict["py"] = self.__py

        sympy_params_dict["gx"] = self.__gx
        sympy_params_dict["gy"] = self.__gy

        sympy_params_dict["ds_basis"] = self.__basis_vec_ds
        sympy_params_dict["lpf_basis"] = self.__basis_vec_lpf

        sympy_params_dict["dsx"] = self.__dsx
        sympy_params_dict["dsy"] = self.__dsy

        sympy_params_dict["cons_x"] = self.__constraints_x
        sympy_params_dict["cons_y"] = self.__constraints_y

        return sympy_params_dict


def main_sos():
    # define a stability obj
    sos = SymbolicPLYDS()
    sos.arrange_constraints()

    # function test for ds
    x = np.array([2, 3])
    px = np.zeros(shape=sos.sos_params["px"].shape)
    py = np.random.rand(*sos.sos_params["py"].shape)
    logger.info(f'DS result: {sos.ds(x, px, py)}')

    # function test for lpf
    qx = np.zeros(shape=sos.sos_params["qx"].shape)
    qy = np.random.rand(*sos.sos_params["qy"].shape)
    logger.info(f'LPF result: {sos.lpf(x, qx, qy)}')

    # function test for lpf_der
    gx = np.zeros(shape=sos.sos_params["gx"].shape)
    gy = np.random.rand(*sos.sos_params["gy"].shape)
    logger.info(f'LPFDER results: {sos.lpf(x, gx, gy)}')


# main entry
if __name__ == "__main__":
    main_sos()