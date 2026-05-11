import tensorlayerx as tlx
import numpy as np


EPS_FLOAT32 = 1e-4
EPS_FLOAT64 = 1e-7

cosh_bound_float32 = 85
cosh_bound_float64 = 700
sinh_bound_float32 = 85
sinh_bound_float64 = 500


def cosh(x):
    x = tlx.minimum(x, tlx.convert_to_tensor(cosh_bound_float32 if x.dtype == tlx.float32 else cosh_bound_float64, dtype=x.dtype))
    return tlx.cosh(x)


def sinh(x):
    x = tlx.minimum(x, tlx.convert_to_tensor(sinh_bound_float32 if x.dtype == tlx.float32 else sinh_bound_float64, dtype=x.dtype))
    return tlx.sinh(x)


def tanh(x):
    return tlx.tanh(x)


def cos(x):
    return tlx.cos(x)


def sin(x):
    return tlx.sin(x)


class SinhDiv:
    @staticmethod
    def forward(x):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y_stable = tlx.ones_like(x)
        y = sinh(x) / x
        return tlx.where(tlx.abs(x) < eps, y_stable, y)

    @staticmethod
    def backward(x, grad_output):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y = (x * cosh(x) - sinh(x)) / x ** 2
        y_stable = tlx.zeros_like(x)
        return tlx.where(tlx.abs(x) < eps, y_stable, y) * grad_output


def sinh_div(x):
    return SinhDiv.forward(x)


class SinhDivCube:
    @staticmethod
    def forward(x):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y_stable = tlx.ones_like(x) / 6
        y = sinh(x) / x ** 3
        return tlx.where(tlx.abs(x) < eps, y_stable, y)

    @staticmethod
    def backward(x, grad_output):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y = (x * cosh(x) - 3 * sinh(x)) / x ** 4
        y_stable = tlx.zeros_like(x)
        return tlx.where(tlx.abs(x) < eps, y_stable, y) * grad_output


def sinh_div_cube(x):
    return SinhDivCube.forward(x)


class CoshDivSquare:
    @staticmethod
    def forward(x):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y_stable = tlx.ones_like(x) * 0.5
        y = cosh(x) / x ** 2
        return tlx.where(tlx.abs(x) < eps, y_stable, y)

    @staticmethod
    def backward(x, grad_output):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y = (x * sinh(x) - 2 * cosh(x)) / x ** 3
        y_stable = tlx.zeros_like(x)
        return tlx.where(tlx.abs(x) < eps, y_stable, y) * grad_output


def cosh_div_square(x):
    return CoshDivSquare.forward(x)


class SinDiv:
    @staticmethod
    def forward(x):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y_stable = tlx.ones_like(x)
        y = tlx.sin(x) / x
        return tlx.where(tlx.abs(x) < eps, y_stable, y)

    @staticmethod
    def backward(x, grad_output):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y = (x * tlx.cos(x) - tlx.sin(x)) / x ** 2
        y_stable = tlx.zeros_like(x)
        return tlx.where(tlx.abs(x) < eps, y_stable, y) * grad_output


def sin_div(x):
    return SinDiv.forward(x)


class CosDivSquare:
    @staticmethod
    def forward(x):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y_stable = -tlx.ones_like(x) * 0.5
        y = tlx.cos(x) / x ** 2
        return tlx.where(tlx.abs(x) < eps, y_stable, y)

    @staticmethod
    def backward(x, grad_output):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y = (-x * tlx.sin(x) - 2 * tlx.cos(x)) / x ** 3
        y_stable = tlx.zeros_like(x)
        return tlx.where(tlx.abs(x) < eps, y_stable, y) * grad_output


def cos_div_square(x):
    return CosDivSquare.forward(x)


class SinDivCube:
    @staticmethod
    def forward(x):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y_stable = -tlx.ones_like(x) / 6
        y = tlx.sin(x) / x ** 3
        return tlx.where(tlx.abs(x) < eps, y_stable, y)

    @staticmethod
    def backward(x, grad_output):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        y = (x * tlx.cos(x) - 3 * tlx.sin(x)) / x ** 4
        y_stable = tlx.ones_like(x) / 24
        return tlx.where(tlx.abs(x) < eps, y_stable, y) * grad_output


def sin_div_cube(x):
    return SinDivCube.forward(x)


class Acosh:
    @staticmethod
    def forward(x):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        x = tlx.maximum(x, tlx.convert_to_tensor(1, dtype=x.dtype) + tlx.convert_to_tensor(eps, dtype=x.dtype))
        z = tlx.sqrt(x * x - 1)
        return tlx.log(x + z)

    @staticmethod
    def backward(x, g):
        eps = EPS_FLOAT32 if x.dtype == tlx.float32 else EPS_FLOAT64
        z = tlx.sqrt(x * x - 1)
        z = tlx.maximum(z, tlx.convert_to_tensor(eps, dtype=z.dtype))
        return g / z


def arcosh(x):
    return Acosh.forward(x)
