import numpy
from theano import tensor
from blocks.bricks import Linear
from blocks.initialization import Constant, IsotropicGaussian


def test_linearlike_subclass_initialize_works_overridden_w():
    class NotQuiteLinear(Linear):
        @property
        def W(self):
            W = super(NotQuiteLinear, self).W
            return W / tensor.sqrt((W ** 2).sum(axis=0))

    brick = NotQuiteLinear(5, 10, weights_init=IsotropicGaussian(0.02),
                           biases_init=Constant(1))
    brick.initialize()
    assert not numpy.isnan(brick.parameters[0].get_value()).any()
    numpy.testing.assert_allclose((brick.W ** 2).sum(axis=0).eval(), 1,
                                  rtol=1e-6)
