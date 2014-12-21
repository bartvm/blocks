__author__ = 'Dmitry Serdyuk'

from abc import ABCMeta, abstractmethod

from theano import function

from blocks.sequence_generators import BaseSequenceGenerator



class Search(object):
    """
    Base class for any search in recurrent network
    """
    __metaclass__ = ABCMeta

    def __init__(self, sequence_generator):
        if not isinstance(sequence_generator, BaseSequenceGenerator):
            raise ValueError("Search needs BaseSequenceGenerator object")
        self.generator = sequence_generator

    @abstractmethod
    def compile(self):
        self.compiled = True

    @abstractmethod
    def search(self):
        if not self.compiled:
            self.compile()


class BeamSearch(Search):
    """
    Class for beam search
    """

    def __init__(self, sequence_generator):
        super(BeamSearch, self).__init__(sequence_generator)

    def compile(self):
        self.initial_state_computer = \
            function(inputs=[self.generator.state_names], # states
                     outputs=[self.generator.inital_state('initial_computer',
                                                         batch_size=1)])
        raise NotImplementedError()

        super(BeamSearch, self).compile()

    def search(self):
        super(BeamSearch, self).search()
        raise NotImplementedError()


class GreedySearch(BeamSearch):
    """
    Greedy search in recurrent network
    """

    def __init__(self):
        raise NotImplementedError()