#!/usr/bin/env python

import numpy as np


class Stack:
    def __init__(self):
        self._items = []

    def __repr__(self):
        return repr([item.shape for item in self])

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, item):
        return self._items[item]

    def push(self, item):
        self._items.append(item)

    def pop(self):
        try:
            return self._items.pop()
        except IndexError:
            return None

    @property
    def top(self):
        return self._items[-1]

    @top.setter
    def top(self, item):
        self._items[-1] = item


class StackMachine(Stack):
    def __init__(self):
        super().__init__()
        self.decode = {
            "NEG": self.neg,
            "ADD": self.add,
            "MUL": self.mul,
            "SUB": self.sub,
            "DIV": self.div,
            "MATMUL": self.matmul,
            "TRANSPOSE": self.transpose,
            "INV": self.inv,
            "LOAD": self.push,
        }

    def neg(self):
        self.top = -self.top

    def add(self):
        x = self.pop()
        self.top += x

    def mul(self):
        x = self.pop()
        self.top *= x

    def sub(self):
        x = self.pop()
        self.top -= x

    def div(self):
        x = self.pop()
        self.top /= x

    def matmul(self):
        x = self.pop()
        self.top = self.top @ x

    def transpose(self):
        self.top = self.top.T

    def inv(self):
        self.top = np.linalg.inv(self.top)

    def execute(self, instructions):
        for op, *args in instructions:
            self.decode[op](*args)
            print(op, self)
        print(self.top)


def main():
    X = np.array([[1, 2, 3], [1, 4, 2], [2, 5, 7], [2, 4, 6], [7, 4, 2]])
    y = np.array([2, 4, 91, 3, 7])
    instructions = [
        ("LOAD", X),
        ("TRANSPOSE",),
        ("LOAD", X),
        ("MATMUL",),
        ("INV",),
        ("LOAD", X),
        ("TRANSPOSE",),
        ("LOAD", y),
        ("MATMUL",),
        ("MATMUL",),
    ]
    m = StackMachine()
    m.execute(instructions)


if __name__ == "__main__":
    main()
