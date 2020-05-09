import json
from abc import ABC, abstractmethod
from numbers import Number

import numpy as np


__all__ = ["new_design_matrix", "new_numpy_array", "expit", "diag", "sum_", "xlogy"]

GOOD = "👍"
BAD = "👎"

mock_registry = dict()
array_registry = dict()


class Mock:
    global mock_registry
    global array_registry
    mocks = mock_registry
    arrays = array_registry

    def __init__(self):
        self.name = None
        self.expr = None
        self.tree = None

    def __repr__(self):
        return self.expr

    @property
    def code(self):
        return self.tree.code


class Scalar(Mock):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr


class Matrix(Mock):
    def __init__(self, shape, name=None, tree=None):
        super().__init__()
        self.shape = shape
        self.expr = name
        self.name = name
        if name:
            self.tree = Terminal(name, shape)
        else:
            self.tree = tree

    @property
    def T(self):
        mat = Matrix(shape=(self.shape[1], self.shape[0]), tree=self.tree)
        mat.expr = f"({self.expr}).T"
        mat.tree = UnaryOp("TRANSPOSE", mat.tree, mat.shape)
        return mat

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        shape = self.shape[:-1] + other.shape[1:]
        mat = Matrix(shape=shape)
        mat.expr = f"({self.expr} @ {other.expr})"
        mat.tree = BinaryOp("MATMUL", [self.tree, other.tree], mat.shape)
        return mat

    def __add__(self, other):
        if not isinstance(other, Matrix) and not isinstance(other, Number):
            raise TypeError(unsupported_operand(self, other))
        mat = Matrix(shape=self.shape)
        if isinstance(other, Matrix):
            assert self.shape == other.shape
            mat.expr = f"({self.expr} + {other.expr})"
            mat.tree = BinaryOp("ADD", [self.tree, other.tree], mat.shape)
        elif isinstance(other, Number):
            mat.expr = f"({self.expr} + {other})"
            mat.tree = BinaryOp("ADD", [self.tree, str(other)], mat.shape)
        return mat

    def __radd__(self, other):
        assert isinstance(other, Matrix) or isinstance(other, Number)
        mat = Matrix(shape=self.shape)
        if isinstance(other, Matrix):
            assert self.shape == other.shape
            mat.expr = f"({other.expr} + {self.expr})"
            mat.tree = BinaryOp("ADD", [other.tree, self.tree], mat.shape)
        elif isinstance(other, Number):
            mat.expr = f"({other} + {self.expr})"
            mat.tree = BinaryOp("ADD", [str(other), self.tree], mat.shape)
        return mat

    def __mul__(self, other):
        assert self.shape == other.shape
        mat = Matrix(shape=self.shape)
        mat.expr = f"({self.expr} * {other.expr})"
        mat.tree = BinaryOp("MUL", [self.tree, other.tree], mat.shape)
        return mat

    def __truediv__(self, other):
        assert self.shape == other.shape
        mat = Matrix(shape=self.shape)
        mat.expr = f"({self.expr} / {other.expr})"
        mat.tree = BinaryOp("DIV", [self.tree, other.tree], mat.shape)
        return mat

    def __neg__(self):
        mat = Matrix(shape=self.shape)
        mat.expr = f"-{self.expr}"
        mat.tree = UnaryOp("UNARY_MINUS", mat.tree, mat.shape)
        return mat

    def __sub__(self, other):
        assert isinstance(other, Matrix) or isinstance(other, Number)
        mat = Matrix(shape=self.shape)
        if isinstance(other, Matrix):
            assert self.shape == other.shape
            mat.expr = f"({self.expr} - {other.expr})"
            mat.tree = BinaryOp("SUB", [other.tree, self.tree], mat.shape)
        elif isinstance(other, Number):
            mat.expr = f"({self.expr} - {other})"
            mat.tree = BinaryOp("SUB", [str(other), self.tree], mat.shape)
        return mat

    def __rsub__(self, other):
        assert isinstance(other, Matrix) or isinstance(other, Number)
        mat = Matrix(shape=self.shape)
        if isinstance(other, Matrix):
            assert self.shape == other.shape
            mat.expr = f"({other.expr} - {self.expr})"
            mat.tree = BinaryOp("SUB", [other.tree, self.tree], mat.shape)
        elif isinstance(other, Number):
            mat.expr = f"({other} - {self.expr})"
            mat.tree = BinaryOp("SUB", [str(other), self.tree], mat.shape)
        return mat

    def len(self):
        if self.name:
            n = Scalar(f"{self.name}.shape[0]")
            return n


def unsupported_operand(op_1, op_2):
    msg = f"unsupported operand type(s) for +: {type(op_1)} and {type(op_2)}"
    return msg


class NumpyArray(Matrix):
    def __init__(self, iterable):
        array = np.array(iterable)
        name = new_array_name()
        super().__init__(array.shape, name)
        self.array = array
        array_registry[name] = array.tolist()


def new_numpy_array(iterable):
    return NumpyArray(iterable)


class NumberObs:
    def __repr__(self):
        return "n"

    def __eq__(self, other):
        if type(self) == type(other):
            return True
        else:
            return False


def new_mock_name():
    return "m_" + str(len(mock_registry))


def new_array_name():
    return "a_" + str(len(array_registry))


def inv(mat):
    shape = mat.shape
    expr = mat.expr
    mat = Matrix(shape=shape)
    mat.expr = f"inv({expr})"
    mat.tree = UnaryOp("INV", mat.tree, mat.shape)
    return mat


def sum_(mat):
    shape = mat.shape
    expr = mat.expr
    mat = Matrix(shape=shape)
    mat.expr = f"np.sum({expr})"
    mat.tree = UnaryOp("SUM", mat.tree, mat.shape)
    return mat


def expit(mat):
    shape = mat.shape
    expr = mat.expr
    mat = Matrix(shape=shape)
    mat.expr = f"scipy.special.expit({expr})"
    mat.tree = UnaryOp("EXPIT", mat.tree, mat.shape)
    return mat


def xlogy(mat_1, mat_2):
    assert mat_1.shape == mat_2.shape
    shape = mat_1.shape
    expr_1 = mat_1.expr
    expr_2 = mat_2.expr
    mat = Matrix(shape=shape)
    mat.expr = f"scipy.special.xlogy({expr_1}, {expr_2})"
    mat.tree = BinaryOp("SUB", [mat_1.tree, mat_2.tree], mat.shape)
    return mat


def diag(mat):
    shape = mat.shape
    assert len(shape) == 1, f"diag expects order-1 tensor, order-{len(shape)} was given"
    expr = mat.expr
    mat = Matrix(shape=(shape[0], shape[0]))
    mat.expr = f"np.diag({expr})"
    mat.tree = UnaryOp("DIAG", mat.tree, mat.shape)
    return mat


class DesignMatrix(Matrix):
    def __init__(self, varnames):
        n_feat = len(varnames.split(" "))
        if n_feat == 1:
            shape = (NumberObs(),)
        else:
            shape = NumberObs(), n_feat
        name = new_mock_name()
        super().__init__(shape, name)
        self.varnames = varnames.split(" ")
        mock_registry[name] = self.varnames


def new_design_matrix(varnames):
    return DesignMatrix(varnames)


class Node(ABC):
    def __init__(self):
        self.name = None
        self.children = None

    def export(self):
        if self.children:
            try:
                return {self.name: [c.export() for c in self.children]}
            except TypeError:
                return {self.name: self.children.export()}
        else:
            return self.name

    def __repr__(self):
        return json.dumps(self.export(), indent=4)

    @property
    @abstractmethod
    def code(self):
        pass


def is_privacy_compliant(shape):
    return BAD if NumberObs() in shape else GOOD


class Terminal(Node):
    def __init__(self, name, shape):
        super().__init__()
        self.name = name
        self.shape = shape

    @property
    def code(self):
        privacy = is_privacy_compliant(self.shape)
        return [f"LOAD {self.name} -> {self.shape}, privacy:{privacy}"]


class UnaryOp(Node):
    def __init__(self, name, child, shape):
        super().__init__()
        self.name = name
        self.children = child
        self.shape = shape

    @property
    def code(self):
        privacy = is_privacy_compliant(self.shape)
        return self.children.code + [f"{self.name} -> {self.shape}, privacy:{privacy}"]


class BinaryOp(Node):
    def __init__(self, name, children, shape):
        super().__init__()
        self.name = name
        self.children = children
        self.shape = shape

    @property
    def code(self):
        privacy = is_privacy_compliant(self.shape)
        return sum([c.code for c in self.children], []) + [
            f"{self.name} -> " f"{self.shape}, privacy:{privacy}"
        ]
