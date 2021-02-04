from __future__ import annotations
# requires at least python 3.7 for annotations

import threading
class Singleton(object):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            with Singleton._instance_lock:
                if not hasattr(Singleton, "_instance"):
                    Singleton._instance = object.__new__(cls)
        return Singleton._instance

from enum import Enum

class SemanticNetsAgent:
    class ConstraintNotMet(Exception):
        pass

    class LeftRight(Enum):
        left = 1
        right = 2
        
        @classmethod
        def opposite_side(cls, side:SemanticNetsAgent.LeftRight):
            return cls.left if side == cls.right else cls.right

    class State:
        # a class that can record the number of sheeps and wolves on both sides
        def __init__(self, left_wolves, left_sheep, right_wolves, right_sheep, shepard_at:SemanticNetsAgent.LeftRight):
            
            self.check_constraints(left_wolves, left_sheep, right_wolves, right_sheep)
            
            self.left_wolves = left_wolves
            self.left_sheep = left_sheep
            self.right_wolves = right_wolves
            self.right_sheep = right_sheep
            self.shepard_at = shepard_at

        @classmethod
        def check_constraints(cls, left_wolves, left_sheep, right_wolves, right_sheep):
            if left_wolves < 0  or left_sheep < 0 or right_wolves < 0 or right_sheep < 0:
                raise SemanticNetsAgent.ConstraintNotMet(f'Number of wolves or sheep < 0. {left_wolves}, {left_sheep}, {right_wolves}, {right_sheep}')
            if left_sheep != 0 and left_wolves > left_sheep:  # does it matter if sheep == 0?
                raise SemanticNetsAgent.ConstraintNotMet(f'{left_wolves} left_wolves have overpowered {left_sheep} left_sheep')
            if right_sheep != 0 and right_wolves > right_sheep:  # does it matter if sheep == 0?
                raise SemanticNetsAgent.ConstraintNotMet(f'{right_wolves} right_wolves have overpowered {right_sheep} right_sheep')

        def __eq__(self, other):
            if isinstance(other, SemanticNetsAgent.State):
                return (self.left_wolves == other.left_wolves and self.left_sheep == other.left_sheep
                    and self.right_wolves == other.right_wolves and self.right_sheep == other.right_sheep
                    and self.shepard_at == other.shepard_at)
            else:
                raise ValueError('You can only compare two SemanticNetsAgent.State objects with ==')
        def __repr__(self):
            return f'State({(self.left_wolves, self.left_sheep, self.right_wolves, self.right_sheep)})'

    class StateNode:
        # a class that wraps a State object, recording its father and son
        def __init__(self, state, depth=0, fathers:list=[], sons:list=[]):
            '''
            :param fathers: list of SemanticNetsAgent.StateNode. Usually you should only specify 1 father in the list when you create a node
            :param sons: list of SemanticNetsAgent.StateNode
            '''
            fathers = list(fathers); sons = list(sons)
            if depth == 0 and fathers:
                raise ValueError('You are building a root node with depth=0 but you specified its father')
            if depth != 0 and not fathers:
                raise ValueError('You must specify a father node of type SemanticNetsAgent.StateNode if the depth is not 0')
            self.state = state
            self.depth = depth
            self.fathers = fathers
            self.sons = sons
            
        def __eq__(self, other):
            if isinstance(other, SemanticNetsAgent.StateNode):
                return other.state == self.state and other.depth == self.depth
            else:
                raise ValueError('You can only compate two SemanticNetsAgent.StateNode objects with ==')
            
        def __repr__(self):
            state = self.state
            return f'StateNode({(state.left_wolves, state.left_sheep, state.right_wolves, state.right_sheep)}, depth={self.depth})'

    class StateTree():
        def __init__(self):
            self.state_tree = []  # expected: list of list

        def __getitem__(self, item):
            return self.state_tree[item]
        def __len__(self):
            return len(self.state_tree)

        def insert_state_node(self, state_node:SemanticNetsAgent.StateNode) -> bool:
            '''
            node inserted into state_tree without state repetition, and with its father-son relationships registered
            :param state_node: SemanticNetsAgent.StateNode. If the depth is not 0, you should specify its only father
            :param depth: The depth at which the state is inserted. The root of StateTree, as the initial input state, has a depth 0
            :returns: whether the node is truely inserted. If a node of the same state is found in the tree, the node will not be inserted
            '''
            assert len(state_node.sons) == 0
            
            max_index = len(self.state_tree) - 1
            if state_node.depth < max_index:
                raise ValueError(f'You are inserting a node with too small depth {state_node.depth}. Current tree depth {max_index}')
            if state_node.depth > max_index + 1:
                raise ValueError(f'You are inserting a node with too large depth {state_node.depth}. Current tree depth {max_index}')
            if state_node.depth == max_index + 1:
                self.state_tree.append([])
                max_index += 1
                
            if state_node.depth == 0:
                assert len(state_node.fathers) == 0
                if self.state_tree[0]:  # there has already been a root
                    raise ValueError('There has already been a root in StateTree')
                self.state_tree[0].append(state_node)
            else:
                assert len(state_node.fathers) == 1
                same_state_node = self.find_state_node_in_tree(state_node)
                if not same_state_node:  # no same state in previous results
                    state_node.fathers[0].sons.append(state_node)
                    self.state_tree[state_node.depth].append(state_node)
                    return True
                # same state_node found in the tree:
                if same_state_node.depth < state_node.depth:
                    # this means that the generated state_node is unproductive
                    return False  # drop the current state_node
                elif same_state_node.depth == state_node.depth:
                    # two fathers from the previous layer generated the same son
                    # we should not insert a new node, but still register the new father-son relationships
                    for father in state_node.fathers:
                        self.insert_node_to_list_without_repetition(same_state_node, father.sons)
                        self.insert_node_to_list_without_repetition(father, same_state_node.fathers)
                    return False
        
        def find_state_node_in_tree(self, state_node:SemanticNetsAgent.StateNode):
            # check whether there is a node of the same state in the tree
            for layer in self.state_tree:
                neighbor_state_node = self.find_state_node_in_list(state_node, layer)
                if neighbor_state_node:
                    return neighbor_state_node
        
        def find_state_node_in_list(self, state_node:SemanticNetsAgent.StateNode, layer:list):
            for neighbor_state_node in layer:
                if neighbor_state_node.state == state_node.state:
                    return neighbor_state_node  # same state found in a layer

        def insert_node_to_list_without_repetition(self, state_node, l:list):
            same_state_node = self.find_state_node_in_list(state_node, l)
            if not same_state_node:
                l.append(state_node)
                return True
            else:
                return False

    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, initial_sheep, initial_wolves):
        #Add your code here! Your solve method should receive
        #the initial number of sheep and wolves as integers,
        #and return a list of 2-tuples that represent the moves
        #required to get all sheep and wolves from the left
        #side of the river to the right.
        #
        #If it is impossible to move the animals over according
        #to the rules of the problem, return an empty list of
        #moves.
        self.state_tree = self.StateTree()
        self.final_node = None
        self.solution_state = self.State(0,0, initial_wolves, initial_sheep, self.LeftRight.right)

        self.shepard_at = self.LeftRight.left
        root_state = self.State(initial_wolves, initial_sheep, 0, 0, self.LeftRight.left)  # initial state assumed to be valid
        root_node = self.StateNode(root_state, depth=0)
        self.state_tree.insert_state_node(root_node)
        
        while 1:
            new_node_inserted = False
            for next_layer_node in self.solution_generator(len(self.state_tree.state_tree)):
                if next_layer_node.state == self.solution_state:
                    self.final_node = next_layer_node
                if self.state_tree.insert_state_node(next_layer_node):
                    new_node_inserted = True
            if self.final_node:
                return self.gen_answer_from_computed_tree()
            if new_node_inserted == False:
                return []
                # for layer in self.state_tree.state_tree:
                #     print(f'depth: {layer[0].depth}')
                #     for state_node in layer:
                #         print(state_node)
                # raise ValueError('No Solution!')
            self.shepard_at = self.LeftRight.opposite_side(self.shepard_at)

    def gen_answer_from_computed_tree(self):
        assert self.final_node is not None
        process = []
        current_node = self.final_node
        while current_node.fathers:
            selected_father = current_node.fathers[0]
            diff = (abs(selected_father.state.left_sheep - current_node.state.left_sheep),
                    abs(selected_father.state.left_wolves - current_node.state.left_wolves), )
            process.insert(0, diff)  # insert diff into the head of process
            current_node = selected_father
        return process
        
    def solution_generator(self, next_tree_depth):
        assert len(self.state_tree.state_tree[-1]) > 0
        
        def single_state_generator(prev_state_node, lw_change, ls_change, rw_change, rs_change, shepard_at):
            try:
                prev_state = prev_state_node.state
                return self.StateNode(
                    self.State(prev_state.left_wolves + lw_change, prev_state.left_sheep + ls_change,
                               prev_state.right_wolves + rw_change, prev_state.right_sheep + rs_change,
                               shepard_at),
                    depth=next_tree_depth, fathers=[prev_state_node]
                )
            except self.ConstraintNotMet as e:
                return None

        if self.shepard_at == self.LeftRight.left:
            for prev_state_node_in_layer in self.state_tree[-1]:
                for lw, ls, rw, rs in [(-2,0,+2,0),(0,-2,0,+2),(-1,0,+1,0),(0,-1,0,+1),(-1,-1,+1,+1)]:
                    new_single_state_node = single_state_generator(prev_state_node_in_layer, lw, ls, rw, rs, self.LeftRight.right)
                    if new_single_state_node:
                        yield new_single_state_node
                
        elif self.shepard_at == self.LeftRight.right:
            for prev_state_node_in_layer in self.state_tree[-1]:
                for lw, ls, rw, rs in [(+2,0,-2,0),(0,+2,0,-2),(+1,0,-1,0),(0,+1,0,-1),(+1,+1,-1,-1)]:
                    new_single_state_node = single_state_generator(prev_state_node_in_layer, lw, ls, rw, rs, self.LeftRight.left)
                    if new_single_state_node:
                        yield new_single_state_node
        else:
            raise ValueError(f'Unexpected state of shepard: {self.shepard_at}')

if __name__ == '__main__':
    # singleton test
    class Tree(Singleton):
        def __init__(self, param=1):
            self.param = param

    a = Tree()
    b = Tree(2)
    print(a, a.param)
    print(b, b.param)
    # a and b has the same address in memory; param == 2
