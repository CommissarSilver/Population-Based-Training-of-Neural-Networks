import random
import numpy as np


class Node:
    def __init__(self, left_node, right_node, content=None, is_leaf=False, index=None):
        self.left_node = left_node
        self.right_node = right_node
        self.is_leaf = is_leaf
        self.content = content
        if not self.is_leaf:
            self.value = self.left_node.value + self.right_node.value
        self.parent = None
        self.index = index
        if left_node is not None:
            left_node.parent = self
        if right_node is not None:
            right_node.parent = self

    @classmethod
    def create_leaf(cls, experience, value, index):
        leaf = cls(None, None, experience, is_leaf=True, index=index)
        leaf.value = value
        return leaf


def create_tree(memory: dict):
    values = [memory[i]['error'] for i in memory.keys()]
    nodes = [Node.create_leaf(memory[index], value, index) for index, value in enumerate(values)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node
    if node.left_node.value >= value:
        return retrieve(value, node.left_node)
    else:
        return retrieve(value - node.right_node.value, node.right_node)


def update(node: Node, new_value: float):
    change = new_value - node.value
    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    node.value += change
    if node.parent is not None:
        propagate_changes(change, node.parent)
