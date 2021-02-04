# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy

from RavensFigure import RavensFigure

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
    def Solve(self,problem):
        self.clear_opened_figures()
        
        solution = 1
        if problem.problemType == '2x2':
            solution = self.solve22(problem, self.get_all_figures(problem))
            # or just skip the problem!
        elif problem.problemType == '3x3':
            solution = self.solve33(problem, self.get_all_figures(problem))
            # or just skip the problem!

        self.clear_opened_figures()
        
        return solution
    
    def solve22(self, problem, figures):
        '''
        available figures: A-C, '1'-'6'
        '''
        return 1

    def solve33(self, problem, figures):
        '''
        available figures: A-H, '1'-'8'
        '''
        return 1

    def get_all_figures(self, problem, figures_str=None):
        '''
        :param figures_str: 'ABC123456'; only open images in figures_str if specified
        '''
        fig_dict = dict()
        if figures_str:
            for char in figures_str:
                fig_dict[char] = self.get_figure(problem.figures[char])
        else:
            for char in problem.figures:
                fig_dict[char] = self.get_figure(problem.figures[char])
        return fig_dict

    def get_figure(self, ravens_figure: RavensFigure):
        f = open(ravens_figure.visualFilename, 'rb')
        self.opened_figures.add(f)
        return Image.open(f)
    
    def clear_opened_figures(self):
        try:
            for f in self.opened_figures:
                f.close()
        except (AttributeError, NameError):  # no varaible self.opened_figures
            pass
        finally:
            self.opened_figures = set()