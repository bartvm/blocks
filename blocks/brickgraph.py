from blocks.graph import ComputationGraph
from blocks.filter import get_brick
from picklable_itertools import chain


class BrickComputationGraph(ComputationGraph):
    """Computation graph annotated with bricks.

    Attributes
    ----------
    top_bricks : list of :class:`~blocks.bricks.base.Brick`
        The top bricks from the computation graph, that is those
        that are not children of other bricks.

    """
    def __init__(self, outputs):
        super(BrickComputationGraph, self).__init__(outputs)
        self._get_bricks()

    def _get_bricks(self):
        bricks = [get_brick(var) for var
                  in self.variables + self.scan_variables if get_brick(var)]
        children = set(chain(*(brick.children for brick in bricks)))
        # Quadratic complexity: we should not have thousands of
        # top-level bricks.
        self.top_bricks = []
        for brick in bricks:
            if brick not in children and brick not in self.top_bricks:
                self.top_bricks.append(brick)
        if len(set(b.name for b in self.top_bricks)) < len(self.top_bricks):
            raise ValueError("top bricks with the same name")
