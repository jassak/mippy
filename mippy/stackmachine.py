#!/usr/bin/env python

import numpy as np

from mippy.expressions import GOOD, BAD


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
    def __init__(self, memory):
        self.mem = memory
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
            "LOAD": self.load,
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

    def load(self, addr):
        self.push(self.mem[addr])

    def execute(self, instructions):
        for op, *args in instructions:
            self.decode[op](*args)
            # print(op, self)
        return self.top


def remote_eval(workers, expr):
    instructions = list(expr.instructions_annotated)
    instructions.reverse()
    master_mem = {}
    master_code = []

    while instructions:
        worker_code = []
        while instructions[-1].privacy == BAD:
            worker_code.append(instructions.pop().instruction)
        worker_code.append(instructions.pop().instruction)
        res = workers.eval_instructions(worker_code, expr.mocks).sum()

        mem_size = len(master_mem)
        master_mem[mem_size] = res
        master_code.append(("LOAD", mem_size))
        while instructions and instructions[-1].privacy == GOOD:
            master_code.append(instructions.pop().instruction)

    machine = StackMachine(master_mem)
    res = machine.execute(master_code)
    return res


def main():
    X = np.array([[1, 2, 3], [1, 4, 2], [2, 5, 7], [2, 4, 6], [7, 4, 2]])
    y = np.array([2, 4, 91, 3, 7])
    memory = {0: X, 1: y}
    instructions = [
        ("LOAD", 0),
        ("TRANSPOSE",),
        ("LOAD", 0),
        ("MATMUL",),
        ("INV",),
        ("LOAD", 0),
        ("TRANSPOSE",),
        ("LOAD", 1),
        ("MATMUL",),
        ("MATMUL",),
    ]
    m = StackMachine(memory=memory)
    res = m.execute(instructions)
    print(res)


if __name__ == "__main__":
    main()
