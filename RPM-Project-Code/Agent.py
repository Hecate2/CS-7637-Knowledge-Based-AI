# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

from __future__ import annotations
from typing import *
from enum import Enum

import random  # just for generating a random answer...

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy as np
import cv2

from RavensFigure import RavensFigure


def show_figure(figures, winname='fig'):
    '''
    For debugging. show_figure(a_list_of_figures, or a_single_figure, or a_SingleShapeDescriptor)
    '''
    if type(figures) is not list:
        figures = [figures]
    for figure in figures:
        if isinstance(figure, SingleShapeDescriptor):
            figure = figure.single_component_figure
        cv2.imshow(winname, figure)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem) -> int:
        
        solution = 1

        '''for answering only a part of problems'''
        solve_problem_type = 'Basic'
        solve_problem_group = 'B'
        solve_problem_num = [i+1 for i in range(12)]

        problem_num = problem.problemSetName[-1], int(problem.name[-2:])

        # if solve_problem_type in problem.problemSetName and problem_num[0] in solve_problem_group and problem_num[1] in solve_problem_num:
        #     pass
        # else:
        #     return solution
        
        if problem.problemType == '2x2':
            solution = self.solve22(problem, self.get_figures(problem))
            # or just skip the problem!
        elif problem.problemType == '3x3':
            solution = self.solve33(problem, self.get_figures(problem))
            # or just skip the problem!

        return solution

    def solve22(self, problem, figures) -> int:
        '''
        available figures: A-C, '1'-'6'
        '''
        def random_answer():
            return random.randint(1,6)

        def score_relationship_similarity(
                relationship_AB: Dict[str:Dict[str:Dict[Agent.Relationship:Set]]], relationship_CD,
                inverse_relationship_AB:Dict[Agent.Relationship:Set],
                inverse_relationship_CD:Dict[Agent.Relationship:Set]) -> int:
            if not inverse_relationship_AB and not inverse_relationship_CD:
                return 2
            score = 0
            for key in inverse_relationship_AB:
                if key in inverse_relationship_CD \
                    and len(inverse_relationship_AB[key]) == len(inverse_relationship_CD[key]):
                    if key == Agent.Disappear:
                        for disappeared_shape_index_AB in inverse_relationship_AB[Agent.Disappear]:
                            if disappeared_shape_index_AB < 0:
                                disappeared_shape_index_AB = -disappeared_shape_index_AB - 1
                                target_figure_name_AB = inverse_relationship_AB['name'][1]
                            else:
                                target_figure_name_AB = inverse_relationship_AB['name'][0]
                            for disappeared_shape_index_CD in inverse_relationship_CD[Agent.Disappear]:
                                if disappeared_shape_index_CD < 0:
                                    disappeared_shape_index_CD = -disappeared_shape_index_CD - 1
                                    target_figure_name_CD = inverse_relationship_CD['name'][1]
                                else:
                                    target_figure_name_CD = inverse_relationship_CD['name'][0]
                                if self.Same.check_same(descriptors[target_figure_name_AB][disappeared_shape_index_AB],
                                                        descriptors[target_figure_name_CD][disappeared_shape_index_CD]):
                                    score += 1
                    elif key == Agent.Symmetry:
                        for tuple_of_shape_index_AB in inverse_relationship_AB[Agent.Symmetry]:
                            for tuple_of_shape_index_CD in inverse_relationship_CD[Agent.Symmetry]:
                                if relationship_AB[tuple_of_shape_index_AB[0]][tuple_of_shape_index_AB[1]] \
                                == relationship_CD[tuple_of_shape_index_CD[0]][tuple_of_shape_index_CD[1]]:
                                    score += 3
                    elif key == Agent.Same:
                        score += 1
                    else:
                        score += 3
            return score
            
        print(problem.name)
        descriptors = {key: SingleShapeDescriptor.build_single_shape_descriptors(figures[key], figure_name=key) for key in figures}
        # print(descriptors)
        relationship_AB, inverse_relationship_AB = self.generate_relationship_between_figures(descriptors['A'], descriptors['B'], 'AB')
        relationship_AC, inverse_relationship_AC = self.generate_relationship_between_figures(descriptors['A'], descriptors['C'], 'AC')
        # print('AB:', relationship_AB)
        # print('AC:', relationship_AC)
        # print('AB inverse:', inverse_relationship_AB)
        # print('AC inverse:', inverse_relationship_AC)

        max_score = 0
        max_score_ans = 1
        for ans_code in '123456':
            relationship_CD, inverse_relationship_CD = self.generate_relationship_between_figures(descriptors['C'], descriptors[ans_code],f'C{ans_code}')
            relationship_BD, inverse_relationship_BD = self.generate_relationship_between_figures(descriptors['B'], descriptors[ans_code],f'B{ans_code}')
            if (not relationship_AB or relationship_CD) and (not relationship_AC or relationship_BD):
                # relationship found
                # print(f'C{ans_code}:', relationship_CD)
                # print(f'B{ans_code}:', relationship_BD)
                # print(f'C{ans_code} inverse:', inverse_relationship_CD)
                # print(f'B{ans_code} inverse:', inverse_relationship_BD)
                score = score_relationship_similarity(relationship_AB, relationship_CD, inverse_relationship_AB, inverse_relationship_CD) \
                      + score_relationship_similarity(relationship_AC, relationship_BD, inverse_relationship_AC, inverse_relationship_BD)
                if score > max_score:
                    max_score = score
                    max_score_ans = int(ans_code)
        # print('Ans:', max_score_ans)
        return max_score_ans

        # print('A,C')
        # relationship = self.find_geometric_relationships_between_single_shapes(descriptors['A'][0], descriptors['C'][0], outer_edge_only=False)
        # print(relationship)
        # print('B,2')
        # relationship = self.find_geometric_relationships_between_single_shapes(descriptors['B'][0], descriptors['2'][0], outer_edge_only=False)
        # print(relationship)
        # print('B,4')
        # relationship = self.find_geometric_relationships_between_single_shapes(descriptors['B'][0], descriptors['4'][0], outer_edge_only=False)
        # print(relationship)
        # print('C,5')
        # relationship = self.find_geometric_relationships_between_single_shapes(descriptors['C'][0], descriptors['5'][0], outer_edge_only=False)
        # print(relationship)
        # return 1

    def solve33(self, problem, figures) -> int:
        '''
        available figures: A-H, '1'-'8'
        '''
        return 1

    def generate_relationship_between_figures(self, fig1_descriptors: List[SingleShapeDescriptor], fig2_descriptors: List[SingleShapeDescriptor], name:str):
        '''
        :param name: which problems are the descriptors from. e.g. 'A1'
        '''
        # {shape_index_in_fig1:{shape_index_in_fig2:{relationship_type:relationship}}}
        relationship_dict = {'name':name}
        # {relationship:set{(shape_index_in_fig1, shape_index_in_fig2)}}
        inverse_relationship_dict = {'name':name}
        shapes_in_1_not_found_relationship_with_2 = set()
        shapes_in_2_found_relationship_with_1 = set()
        for index1, single_shape_descriptor1 in enumerate(fig1_descriptors):
            relationship_dict[index1] = dict()
            for index2, single_shape_descriptor2 in enumerate(fig2_descriptors):
                if index2 not in shapes_in_2_found_relationship_with_1:
                    relationships = self.find_geometric_relationships_between_single_shapes(single_shape_descriptor1, single_shape_descriptor2)
                    if relationships:
                        relationship_dict[index1][index2] = relationships
                        shapes_in_2_found_relationship_with_1.add(index2)
                        for relationship in relationships:
                            if not inverse_relationship_dict.get(relationship):
                                inverse_relationship_dict[relationship] = set()
                            inverse_relationship_dict[relationship].add((index1, index2))
            if not relationship_dict[index1]:
                shapes_in_1_not_found_relationship_with_2.add(index1)
                del relationship_dict[index1]
        if relationship_dict:  # only find disappering relationships when other relationships are found
            for shape_index_1 in shapes_in_1_not_found_relationship_with_2:
                relationship_dict[shape_index_1] = {Agent.Disappear: True}
                if not inverse_relationship_dict.get(Agent.Disappear):
                    inverse_relationship_dict[Agent.Disappear] = set()
                inverse_relationship_dict[Agent.Disappear].add(shape_index_1)
            for shape_index_2 in range(len(fig2_descriptors)):
                if shape_index_2 not in shapes_in_2_found_relationship_with_1:
                    relationship_dict[-shape_index_2-1] = {Agent.Disappear: True}
                    if not inverse_relationship_dict.get(Agent.Disappear):
                        inverse_relationship_dict[Agent.Disappear] = set()
                    inverse_relationship_dict[Agent.Disappear].add(-shape_index_2-1)
        return relationship_dict, inverse_relationship_dict

    def find_geometric_relationships_between_single_shapes(self,
            descriptor1: SingleShapeDescriptor, descriptor2: SingleShapeDescriptor, outer_edge_only=False,
            stop_when_found=True):
        relationships = dict()
    
        def create_key_value_filtering_none(key, value, filter_func=lambda x: x):
            if filter_func(value):
                relationships[key] = value
                if stop_when_found:
                    raise KeyboardInterrupt('Relationship found')
    
        try:
            # TODO: same does not rule out symmetry, but rules out filling
            create_key_value_filtering_none(self.Same,
                self.Same.check_same(descriptor1, descriptor2, outer_edge_only=outer_edge_only))
            create_key_value_filtering_none(self.Symmetry,
                self.Symmetry(descriptor1, descriptor2, outer_edge_only=outer_edge_only),
                filter_func=lambda x:x.symmetry_types)
            create_key_value_filtering_none(self.Filling,
                self.Filling(descriptor1, descriptor2),
                filter_func=lambda x: x if x.filled_contours_1 or x.filled_contours_2 else None)
        except KeyboardInterrupt:
            pass
        finally:
            return relationships if relationships else None

    @staticmethod
    def traverse_selected_relationships(relationships: Dict[Relationship:Set[Relationship]],
                                        selection: Set[Relationship] = None):
        '''
        This method is likely to be deprecated
        :param relationships: refer to the return value of function find_geometric_relationships_between_single_shapes
        :param selection: which type(s) of relationships to yield. Yield all if None
        '''
        if selection:
            if isinstance(selection, Agent.Relationship):  # selection = Symmetry -> {Symmetry}
                selection = {selection}
            for relationship in relationships[selection]:
                yield relationship
        else:
            # yield all relationships
            for relationship_set in relationships.values():
                for relationship in relationship_set:
                    yield relationship

    def get_figures(self, problem, figures_str=None):
        '''
        :param figures_str: 'ABC123456'; only open images in figures_str if specified
        '''
        fig_dict = dict()
        figures_str = figures_str if figures_str else problem.figures
        for char in figures_str:
            fig_dict[char] = cv2.imread(problem.figures[char].visualFilename, cv2.IMREAD_GRAYSCALE)
        return fig_dict

    class Morphology:
        @staticmethod
        def morphological_difference(im1: np.ndarray, im2: np.ndarray,
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))) -> np.ndarray:
            return cv2.morphologyEx(SingleShapeDescriptor.absolute_difference(im1, im2), cv2.MORPH_OPEN, kernel)
        
        @staticmethod
        def image_soft_equal(im1: np.ndarray, im2: np.ndarray,
                             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), threshold = 1/2500) -> bool:
            '''
            this function eliminates minor differences between two images using the open operation,
            before judging whether they are morphologically equal
            :param kernel: by default, firse erode by 2 pixels and then dilate by 2 pixels
            :param threshold: There may still be some difference after the open operation.
                Neglect the percentage of count of pixel difference under the threshold
            '''
            return np.sum(cv2.morphologyEx(SingleShapeDescriptor.absolute_difference(im1, im2), cv2.MORPH_OPEN, kernel)) / im1.size / np.max(im1) < threshold
        
        @staticmethod
        def draw_edge_of_single_contour(contour, canvas_shape=(184, 184), color=255, thickness=3):
            canvas = np.zeros(canvas_shape, dtype=np.uint8)
            cv2.drawContours(canvas, [contour], -1, color, thickness)
            return canvas
        
        @staticmethod
        def have_hard_intersection(fig1, fig2):
            '''
            :param fig1: np.ndarray or SingleShapeDescriptor
            :param fig2: Same type and shape as fig1
            :return: Whether there is at least one point where the values of figure 1 and figure 2 are both non-zero
            '''
            if isinstance(fig1, SingleShapeDescriptor):
                fig1, fig2 = fig1.single_component_figure, fig2.single_component_figure
            return True if True in np.logical_and(fig1, fig2) else False

    class ContourMatching:
        def __init__(self, fig1: SingleShapeDescriptor, fig2: SingleShapeDescriptor, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))):
            self.fig1, self.fig2 = fig1, fig2
            self.match_dict_1, self.match_dict_2 = dict(), dict()
            for idx1, drawn_contour1 in enumerate(fig1.drawn_contours):
                for idx2, drawn_contour2 in enumerate(fig2.drawn_contours):
                    if Agent.Morphology.image_soft_equal(drawn_contour1, drawn_contour2, kernel):
                        self.match_dict_1[idx1] = idx2
                        self.match_dict_2[idx2] = idx1
            self.disappering_contours_1 = set(filter(lambda idx: idx not in self.match_dict_1, [idx for idx in range(len(fig1.drawn_contours))]))
            self.disappering_contours_2 = set(filter(lambda idx: idx not in self.match_dict_1, [idx for idx in range(len(fig2.drawn_contours))]))

        def __repr__(self):
            return f'ContourMatching: {self.match_dict_1}'

    class Relationship:
        pass

    class Position(Relationship, Enum):
        undetermined = 0
        
        up = 1
        down = 2
        left = 3
        right = 4
    
        upper_left = 5
        upper_right = 6
        lower_left = 7
        lower_right = 8
    
        inner = 9
        outer = 10
        
        @classmethod
        def compare_relative_position_of_two_points(cls, coordinate1, coordinate2, threshold=5) -> Agent.Position:
            '''
            What direction is coordinate 2 in, compared to coordinate 1?
            :param coordinate1: (x,y) tuple or np.ndarray. (0,0) is at top-left of the image.
            :param coordinate2: same as coordinate1.
            :param threshold: how many degrees of grace is allowed?
            For example, threshold==3. coordinate2 is considered to the right of coordinate1
            if its in -3 degrees to +3 degrees to the right of coordinate1
            '''
            threshold = abs(threshold)
            if threshold >= 22.5:
                raise ValueError(f'Too large threshold {threshold}')
            vec = (coordinate2[0] - coordinate1[0], coordinate2[1] - coordinate1[1])
            argument_angle = AnalyticGeometry.argument_angle_degree(vec)
            if -threshold <= argument_angle <= threshold:
                return cls.right
            if -threshold+45 <= argument_angle <= threshold+45:
                return cls.upper_right
            if -threshold+90 <= argument_angle <= threshold+90:
                return cls.up
            if -threshold+135 <= argument_angle <= threshold+135:
                return cls.upper_left
            if -threshold+180 <= argument_angle <= threshold+180:
                return cls.left
            if -threshold+225 <= argument_angle <= threshold+225:
                return cls.lower_left
            if -threshold+270 <= argument_angle <= threshold+270:
                return cls.down
            if -threshold+315 <= argument_angle <= threshold+315:
                return cls.lower_right
            return cls.undetermined
        
        @classmethod
        def compare_relative_position_of_outer_edge_center(cls, fig1: SingleShapeDescriptor, fig2: SingleShapeDescriptor):
            return cls.compare_relative_position_of_two_points(fig1.center_outer_edge, fig2.center_outer_edge)

    class Same(Relationship):
        @staticmethod
        def check_same(fig1, fig2, outer_edge_only=False):
            '''
            :param fig1: np.ndarray or SingleShapeDescriptor object
            :param fig2: same type as fig1
            :param outer_edge_only: if True and figs are SingleShapeDescriptor objects, judge if the outer contours of the figures are symmetric
            '''
            # compare the figures if they are raw np.ndarray; compare edges if they are SingleShapeDescriptor and ignore_filled==True
            is_single_shape_descriptor = isinstance(fig1, SingleShapeDescriptor)
            if outer_edge_only and is_single_shape_descriptor:
                return Agent.Morphology.image_soft_equal(fig1.drawn_contours[0], fig2.drawn_contours[0])
            if is_single_shape_descriptor:
                fig1, fig2 = fig1.single_component_figure, fig2.single_component_figure
            return Agent.Morphology.image_soft_equal(fig1, fig2)

    class Symmetry(Relationship):
        class SymmetryType(Enum):
            up_down = 1
            left_right = 2
            up_down_left_right = 3
            def __repr__(self):
                return f"<{self._name_}>"

        @staticmethod
        def check_up_down_symmetry(fig1, fig2, outer_edge_only=False, flip_direction=0):
            '''
            :param fig1: np.ndarray or SingleShapeDescriptor object
            :param fig2: same type as fig1
            :param outer_edge_only: if True and figs are SingleShapeDescriptor objects, judge if the outer contours of the figures are symmetric
            :param flip_direction: direction parameter for cv2.flip. Should be 0 here
            '''
            # compare the figures if they are raw np.ndarray; compare edges if they are SingleShapeDescriptor and ignore_filled==True
            is_single_shape_descriptor = isinstance(fig1, SingleShapeDescriptor)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # firse erode by 2 pixels and then dilate by 2 pixels
            if outer_edge_only and is_single_shape_descriptor:
                return Agent.Morphology.image_soft_equal(fig1.drawn_contours[0], cv2.flip(fig1.drawn_contours[0], flip_direction))
            if is_single_shape_descriptor:
                fig1, fig2 = fig1.single_component_figure, fig2.single_component_figure
            return Agent.Morphology.image_soft_equal(fig1, cv2.flip(fig2, flip_direction))
        @classmethod
        def check_left_right_symmetry(cls, fig1: np.array, fig2: np.array, outer_edge_only=True, flip_direction=1):
            return cls.check_up_down_symmetry(fig1, fig2, outer_edge_only=outer_edge_only, flip_direction=flip_direction)
        @classmethod
        def check_up_down_left_right_symmetry(cls, fig1: np.array, fig2: np.array, outer_edge_only=True, flip_direction=-1):
            return cls.check_up_down_symmetry(fig1, fig2, outer_edge_only=outer_edge_only, flip_direction=flip_direction)

        @classmethod
        def detect_symmetry(cls, fig1, fig2, outer_edge_only=True) -> Tuple[Agent.Symmetry.SymmetryType]:
            '''
            :param fig1: np.ndarray or SingleShapeDescriptor object
            :param fig2: same type as fig1
            '''
            symmetry = []
            if cls.check_up_down_symmetry(fig1, fig2, outer_edge_only=outer_edge_only):
                symmetry.append(cls.SymmetryType.up_down)
            if cls.check_left_right_symmetry(fig1, fig2, outer_edge_only=outer_edge_only):
                symmetry.append(cls.SymmetryType.left_right)
            if cls.check_up_down_left_right_symmetry(fig1, fig2, outer_edge_only=outer_edge_only):
                symmetry.append(cls.SymmetryType.up_down_left_right)
            return tuple(symmetry) if symmetry else None
        
        def __init__(self, fig1, fig2, outer_edge_only=True):
            symmetry_types = self.detect_symmetry(fig1, fig2, outer_edge_only=outer_edge_only)
            self.symmetry_types = set(symmetry_types) if symmetry_types else None
            
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                raise ValueError(f'You can only compare two {self.__class__.__name__} objects with ==')
            return True if self.symmetry_types & other.symmetry_types else False
        def __repr__(self):
            return f"{self.symmetry_types}"


    class Disappear(Relationship):
        @staticmethod
        def detect_disappear(fig1:SingleShapeDescriptor, fig2_descriptor_list:List[SingleShapeDescriptor]):
            prev_figure_name = fig2_descriptor_list[0].figure_name
            for descriptor in fig2_descriptor_list[1:]:
                if descriptor.figure_name != prev_figure_name:
                    raise ValueError('It seems descriptors from fig2_descriptor_list are not from the same figure')
            for descriptor in fig2_descriptor_list:
                if Agent.Same.check_same(fig1, descriptor):
                    return False
            return True

    class Filling(Relationship):
        def __init__(self, fig1:SingleShapeDescriptor, fig2:SingleShapeDescriptor, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))):
            '''
            Record whether an empty area in fig1 is filled in fig2, or vice versa
            :returns: a set of indexes of contours which are not filled in fig1 but filled in fig2, and vice versa
            '''
            '''WARNING: I do not think this function is fully reliable, thought it works on all the Basic B problems'''
            self.filled_contours_1, self.filled_contours_2 = set(), set()
            if not Agent.Morphology.have_hard_intersection(fig1, fig2):
                # There is no intersection between the two connected components
                return

            difference = Agent.Morphology.morphological_difference(fig1.single_component_figure, fig2.single_component_figure, kernel)
            descriptors_of_difference = SingleShapeDescriptor.build_single_shape_descriptors(difference, reverse_color=False, figure_name=f'AbsDiff between: {fig1.figure_name} {fig2.figure_name}')

            def register_filled_contours(fig, filled_contours):
                if not fig.filled:
                    contour_matchings = [Agent.ContourMatching(fig, descriptor_of_difference, kernel=kernel)
                                         for descriptor_of_difference in descriptors_of_difference]
                    for contour_matching in contour_matchings:
                        for key in contour_matching.match_dict_1:
                            filled_contours.add(key)
            register_filled_contours(fig1, self.filled_contours_1)
            register_filled_contours(fig2, self.filled_contours_2)

        def __repr__(self):
            if self.filled_contours_1 or self.filled_contours_2:
                return f'Index of filled contour: in fig1:{self.filled_contours_1 if self.filled_contours_1 else None}; in fig2:{self.filled_contours_2 if self.filled_contours_2 else None}'
            else:
                return f'No filling relationship'

    # TODO: shape changed

class AnalyticGeometry:
    @staticmethod
    def argument_angle_degree(vector):
        '''
        find the argument angle between a vector and the vector (1,0). in degrees, ranging from 0 to 360
        :param vector: (x,y), np.ndarray or tuple
        '''
        assert len(vector) == 2
        ans = vector[0]/np.linalg.norm(vector)/np.pi*180
        if vector[1] <= 0:
            ans = 360 - ans
        return ans
    
class SingleShapeDescriptor:
    @staticmethod
    def absolute_difference(fig1: np.ndarray, fig2: np.ndarray):
        '''
        The absolute difference of two np.uint8 arrays. 255-0 == 0-255 == 255, 255-255 == 0-0 == 0
        '''
        difference = fig1 - fig2
        difference[difference!=0] = 255
        return difference
    
    @staticmethod
    def detect_contours(figure, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
        contours, hierarchy = cv2.findContours(figure, mode, method)
        return contours, hierarchy

    @staticmethod
    def judge_num_of_edges(contour, perimeter=None, accuracy=0.02) -> int:
        if not perimeter:
            perimeter = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, perimeter * accuracy, True)
        return len(approx)
    
    @staticmethod
    def judge_filled(contours_of_connected_component) -> bool:
        l = len(contours_of_connected_component)
        if l >= 2:
            return False
        # if l > 2:
        #     raise ValueError('It is somehow weird that there are more that 2 contours in a single connected component')
        if l == 1:
            return True
        raise ValueError('It seems the contour contains nothing')
    
    @staticmethod
    def extract_single_connected_component(labels: np.array, label: int) -> np.array:
        '''
        :param labels: a figure on which each single connected component is labeled a different uint8 number
        :param label: which number of connected component would you like to extract?
        :return: extracted figure containing only a single connected component
        '''
        mask = np.array(labels, dtype=np.uint8)
        mask[labels != label] = 0
        mask[labels == label] = 255
        return mask

    @classmethod
    def build_single_shape_descriptors(cls, figure: np.array, figure_name: str = None, reverse_color=True) -> List[SingleShapeDescriptor]:
        '''
        Major object factory to intake a raw input figure. The input figure must take the operation 255-figure.
        :param reverse_color: If the figure is raw from the problems, we should take figure = 255-figure
        :param figure: a raw input figure from the problem
        :return: list of SingleShapeDetector objects, each describing a single connected compoent in the figure
        '''
        if reverse_color:
            figure = 255-figure  # opencv identifies white objects as important ones, while the black serve as backgroud
        ret, labels = cv2.connectedComponents(figure)
        component_descriptions = [cls(cls.extract_single_connected_component(labels, label), figure_name=figure_name) for label in range(1,ret)]
        return component_descriptions

        '''codes to show the images and contours for debugging'''
        # contours, hierarchy = cls.detect_contours(figure)

        # cv2.imshow('component', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # rgb_figure = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(rgb_figure, contours, -1, (0, 255, 0), 2)
        # cv2.imshow("draw", rgb_figure)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    def __init__(self, single_component_figure, figure_name: str = None):
        '''
        :param single_component_figure: a figure containing a single connected component. The values of the component is 255, and the background value is 0
        '''
        self.figure_name = figure_name
        # if a connected component is not filled, contours[0] should be the outer contour
        self.contours, self.hierarchy = self.detect_contours(single_component_figure)
        contours = self.contours
        self.drawn_contours = [Agent.Morphology.draw_edge_of_single_contour(contour, single_component_figure.shape) for contour in contours]
        self.filled = self.judge_filled(contours)
        # if self.filled:
        #     self.single_component_figure = single_component_figure
        # else:  # always fill the unfilled contours
        #     self.single_component_figure = cv2.fillPoly(single_component_figure, [contours[0]], 255)
        self.single_component_figure = single_component_figure

        contour = contours[0]
        M = cv2.moments(contour)
        self.perimeter_outer_edge = cv2.arcLength(contour,True)
        self.count_outer_edge = self.judge_num_of_edges(contour, self.perimeter_outer_edge)
        self.area_outer_edge = abs(M["m00"])
        self.moments_outer_edge = M
        self.center_outer_edge = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # the position of the center
        
        if not self.filled:
            contours = contours[1:]
            M = [cv2.moments(contour) for contour in contours]
            self.perimeter_inner_edges = [cv2.arcLength(contour, True) for contour in contours]
            self.count_inner_edges = [self.judge_num_of_edges(contour, perimeter_inner_edge) for perimeter_inner_edge in self.perimeter_inner_edges]
            self.area_inner_edges = [abs(Mi["m00"]) for Mi in M]
            self.moments_inner_edges = M
            self.center_inner_edges = [(int(Mi["m10"] / Mi["m00"]), int(Mi["m01"] / Mi["m00"])) for Mi in M]

    @property
    def size(self):
        return self.single_component_figure.size

    def __eq__(self, other, perimeter_error_limit=0.01, area_error_limit=0.01):
        '''
        Whether two shapes are "the same" in perimeter, area and count of edges.
        Repositioning and rotation do not affect equality
        '''
        if not isinstance(other, self.__class__):
            raise ValueError(f'You can only compare two {self.__class__.__name__} objects with ==')
        if (
            self.count_outer_edge == other.count_outer_edge and
            abs(self.perimeter_outer_edge - other.perimeter_outer_edge) < perimeter_error_limit * self.perimeter_outer_edge and
            abs(self.area_outer_edge - other.area_outer_edge) < area_error_limit * self.area_outer_edge
        ):
            return True
        else:
            return False

    def __sub__(self, other) -> np.ndarray:
        if not isinstance(other, self.__class__):
            raise ValueError(f'You can only __sub__ two {self.__class__.__name__} objects with "-"')
        return self.absolute_difference(self.single_component_figure, other.single_component_figure)

    def __repr__(self):
        return f'SingleShape from {self.figure_name}: {self.count_outer_edge} edges; {self.area_outer_edge} area; Center={self.center_outer_edge}; Filled={self.filled}'
    