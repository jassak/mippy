from abc import ABC, abstractmethod
import json
import pprint

__all__ = ["new_design_matrix"]


class DesignMatrix:
    registry = dict()

    def __init__(self, varnames):
        self.name = new_name()
        self.registry[self.name] = varnames.split(" ")

    def __repr__(self):
        return self.name

    @staticmethod
    def _new_design_matrix(varnames):
        dm = DesignMatrix(varnames)
        name = dm.name
        shape = NumberObs(), len(dm.registry[name])
        return Matrix(shape, name)


new_design_matrix = DesignMatrix._new_design_matrix


class Matrix:
    global DesignMatrix
    registry = DesignMatrix.registry

    def __init__(self, shape, name=None, tree=None):
        self.shape = shape
        self.expr = name
        if name:
            self.tree = Terminal(name)
        else:
            self.tree = tree

    @property
    def code(self):
        return self.tree.code

    def __repr__(self):
        return self.expr

    @property
    def T(self):
        mat = Matrix(shape=(self.shape[1], self.shape[0]), tree=self.tree)
        mat.expr = self.expr + ".T"
        mat.tree = UnaryOp("TRANSPOSE", mat.tree)
        return mat

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        mat = Matrix(shape=(self.shape[0], other.shape[1]))
        mat.expr = "(" + self.expr + " @ " + other.expr + ")"
        mat.tree = BinaryOp("MATMUL", [self.tree, other.tree])
        return mat

    def __add__(self, other):
        assert self.shape == other.shape
        mat = Matrix(shape=self.shape)
        mat.expr = "(" + self.expr + " + " + other.expr + ")"
        mat.tree = BinaryOp("ADD", [self.tree, other.tree])
        return mat

    def __mul__(self, other):
        assert self.shape == other.shape
        mat = Matrix(shape=self.shape)
        mat.expr = "(" + self.expr + " * " + other.expr + ")"
        mat.tree = BinaryOp("MUL", [self.tree, other.tree])
        return mat


class NumberObs:
    def __repr__(self):
        return "n"

    def __eq__(self, other):
        if type(self) == type(other):
            return True
        else:
            return False


class Node(ABC):
    def __init__(self):
        self.name = None
        self.children = None

    def export(self):
        if self.children:
            try:
                return {self.name: [c.export() for c in self.children]}
            except:
                return {self.name: self.children.export()}
        else:
            return self.name

    def __repr__(self):
        return json.dumps(self.export(), indent=4)

    @property
    @abstractmethod
    def code(self):
        pass


class Terminal(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @property
    def code(self):
        return [f"LOAD {self.name}"]


class UnaryOp(Node):
    def __init__(self, name, child):
        super().__init__()
        self.name = name
        self.children = child

    @property
    def code(self):
        return self.children.code + [f"{self.name}"]


class BinaryOp(Node):
    def __init__(self, name, children):
        super().__init__()
        self.name = name
        self.children = children

    @property
    def code(self):
        return sum([c.code for c in self.children], []) + [f"{self.name}"]


def new_name():
    global DesignMatrix
    return "t_" + str(len(DesignMatrix.registry))


def inv(mat):
    mat.expr = "inv(" + mat.expr + ")"
    mat.tree = UnaryOp("INV", mat.tree)
    return mat


# X = new_design_matrix("v1 v2 v3 v4")
# y = new_design_matrix("aaa")
# S = inv(X.T @ X) @ (X.T @ y)
