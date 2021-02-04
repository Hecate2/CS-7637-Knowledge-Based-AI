from __future__ import annotations

from copy import deepcopy
from enum import Enum
from queue import Queue

class BlockWorldAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    class BlockPosition:
        def __init__(self, block_world_agent: BlockWorldAgent, block_name, stack_index: int, depth: int, underneath_block: str):
            '''
            Records where a block is
            :param index_of_stack: for [["A", "B", "C"], ["D", "E"]], "A" is in stack 0
            :param depth: for [["A", "B", "C"], ["D", "E"]], "A" is at depth 2, on the table
            '''
            self.block_name = block_name
            self.stack_index = stack_index
            self.depth = depth
            if block_name == 'Table':
                # raise ValueError('Table should not be viewed as a block')
                self.stack = []
                self.underneath_block = None
                self.on_table = False
            else:
                self.stack = block_world_agent.state[stack_index]
                self.underneath_block = underneath_block
                self.on_table = True if underneath_block == 'Table' else False
        def __repr__(self):
            if self.block_name == 'Table':
                return f'BlockPosition: "{self.block_name}" with stack_index {self.stack_index}'
            return f'BlockPosition: Block "{self.block_name}" in stack {self.stack} at depth {self.depth} on "{self.underneath_block}"'

    class Subgoal:
        '''The goal of putting a block onto another'''
        def __init__(self, block_world_agent: BlockWorldAgent, block_name: str, onto_block: str):
            self.block_world_agent = block_world_agent
            self.block_name = block_name
            self.onto_block = onto_block
            if onto_block != 'Table':
                self.moveable_subgoals = {BlockWorldAgent.MovableSubgoal(block_world_agent, block_name),
                                          BlockWorldAgent.MovableSubgoal(block_world_agent, onto_block), }
            else:
                self.moveable_subgoals = {BlockWorldAgent.MovableSubgoal(block_world_agent, block_name)}
        @property
        def ready(self):
            score = self.moveable_subgoals_score
            length = len(self.moveable_subgoals)
            if score == length:
                if self.onto_block == 'Table':
                    return True
                if self.onto_block in self.block_world_agent.in_position_blocks:
                    return True
            return False
        
        @property
        def moveable_subgoals_score(self):
            score = 0
            for moveable_subgoal in self.moveable_subgoals:
                if moveable_subgoal.fulfilled:
                    score += 1
            return score
        
        def __repr__(self):
            return f'Subgoal: Move "{self.block_name}" onto "{self.onto_block}". score = {self.moveable_subgoals_score}, ready = {self.ready}.'
        
    class MovableSubgoal:
        '''Whether a block is moveable'''
        def __init__(self, block_world_agent: BlockWorldAgent, block_name: str):
            self.block_world_agent = block_world_agent
            self.block_name = block_name
        @property
        def fulfilled(self):
            block_position = self.block_world_agent.search_block(self.block_name)
            if block_position.depth == 0:
                return True
            else:
                return False
            
        def __repr__(self):
            return f'MovableSubgoal: Make {self.block_name} movable. Fulfilled = {self.fulfilled}'

    def search_block(self, block_name="A") -> BlockPosition:
        if block_name == 'Table':
            return self.BlockPosition(self, 'Table', len(self.state), -1, None)
        for stack_index, stack in enumerate(self.state):
            try:
                block_index = stack.index(block_name)
                return self.BlockPosition(self, block_name, stack_index,
                                          len(stack) - 1 - block_index,
                                          "Table" if block_index == 0 else self.state[stack_index][block_index - 1])
            except ValueError:
                pass
        raise ValueError(f'block "{block_name}" not found')

    def solve(self, initial_arrangement, goal_arrangement):
        # Add your code here! Your solve method should receive
        # as input two arrangements of blocks. The arrangements
        # will be given as lists of lists. The first item in each
        # list will be the bottom block on a stack, proceeding
        # upward. For example, this arrangement:
        #
        # [["A", "B", "C"], ["D", "E"]]
        #
        # ...represents two stacks of blocks: one with B on top
        # of A and C on top of B, and one with E on top of D.
        #
        # Your goal is to return a list of moves that will convert
        # the initial arrangement into the goal arrangement.
        # Moves should be represented as 2-tuples where the first
        # item in the 2-tuple is what block to move, and the
        # second item is where to put it: either on top of another
        # block or on the table (represented by the string "Table").
        #
        # For example, these moves would represent moving block B
        # from the first stack to the second stack in the example
        # above:
        #
        # ("C", "Table")
        # ("B", "E")
        # ("C", "A")
        self.state = deepcopy(initial_arrangement)
        self.goal = deepcopy(goal_arrangement)
        
        # build description for the final goal
        self.build_goal_description()
        
        moves = []
        while self.not_in_position_blocks:
            move, new_state = self.execute_move()
            moves.append(move)
            self.state = new_state
        return moves
        
    def build_goal_description(self):
        self.in_position_blocks = set()
        self.not_in_position_blocks = set()
        self.subgoals = set()
        for goal_stack in self.goal:
            goal_prev_block = 'Table'
            lower_block_in_position = True
            for goal_block in goal_stack:
                if lower_block_in_position:
                    current_block_position = self.search_block(goal_block)
                    if current_block_position.underneath_block == goal_prev_block:
                        self.in_position_blocks.add(goal_block)
                    else:
                        lower_block_in_position = False
                        self.not_in_position_blocks.add(goal_block)
                        self.subgoals.add(BlockWorldAgent.Subgoal(self,goal_block,goal_prev_block))
                else:
                    self.not_in_position_blocks.add(goal_block)
                    self.subgoals.add(BlockWorldAgent.Subgoal(self, goal_block, goal_prev_block))
                goal_prev_block = goal_block

    def generate_move(self):
        for source_stack in self.state:
            if source_stack[-1] in self.in_position_blocks:
                continue
            if len(source_stack) == 1:
                continue
            if len(source_stack) > 1:
                yield (source_stack[-1], 'Table')
            # No need to move a block to another stack
            # for target_stack in self.state:
            #     if source_stack[-1] != target_stack[-1]:
            #         yield (source_stack[-1], target_stack[-1])
    
    def deduce_state_after_move(self, move):
        source, dest = move
        new_state = deepcopy(self.state)
        source_position = self.search_block(source); assert source_position.depth == 0
        new_state[source_position.stack_index].pop()
        if dest == 'Table':
            new_state.append([source])
            return new_state
        dest_position = self.search_block(dest); assert dest_position.depth == 0
        new_state[dest_position.stack_index].append(source)
        if not new_state[source_position.stack_index]:
            new_state.pop(source_position.stack_index)
        return new_state
        
    def mark_block_in_position(self, subgoal):
        self.subgoals.remove(subgoal)
        self.not_in_position_blocks.remove(subgoal.block_name)
        self.in_position_blocks.add(subgoal.block_name)
    
    def execute_move(self):
        for subgoal in self.subgoals:  # is it possible to move a block into its final position?
            if subgoal.ready:
                move = (subgoal.block_name, subgoal.onto_block)
                new_state = self.deduce_state_after_move(move)
                self.mark_block_in_position(subgoal)
                return move, new_state

        # is it possible to move a block to achieve MovableSubgoals?
        max_score = -1
        moveable_subgoals = set()
        for subgoal in self.subgoals:
            for moveable_subgoal in subgoal.moveable_subgoals:
                moveable_subgoals.add(moveable_subgoal)

        def score_move_with_movable_subgoal(source):
            score = 0
            block_position = self.search_block(source)
            for moveable_subgoal in moveable_subgoals:
                if moveable_subgoal.block_name in block_position.stack:
                    score += 1
            return score

        for move in self.generate_move():
            source, dest = move
            score = score_move_with_movable_subgoal(source)
            if score > max_score:
                max_score = score
                planned_move = move
        new_state = self.deduce_state_after_move(planned_move)
        return planned_move, new_state
            