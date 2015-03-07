import numpy
import theano
import logging
from theano import tensor
from blocks.model import Model
from blocks.bricks import Linear, Tanh
from blocks.bricks.cost import SquaredError, CTC
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, SimpleExtension
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import OUTPUT
from matplotlib import pyplot
import sys
import cPickle as pickle

#class MonitorOutputs(SimpleExtension):
#    def __init__(self, **kwargs):
#        super(MonitorOutputs, self).__init__(**kwargs)
#
#    def do(self, *args):
#        cg = ComputationGraph(self.main_loop.algorithm.cost)
#        rnn_out = VariableFilter(application=self.main_loop.model.top_bricks[0].apply,
#                                 roles=[OUTPUT])(cg.variables)[-1]
#        print "NOW: " + str(rnn_out.get_value())


def main(seq_u, seq_y, mask_inputs, mask_labels, hidden_dim, num_classes, n_epochs):
    # Building Model
    u = tensor.tensor3('input_sequences')
    i_mask = tensor.matrix('mask_inputs', dtype='int32')
    l_mask = tensor.matrix('mask_labels', dtype='int32')

    input_to_state = Linear(name='input_to_state',
                            input_dim=seq_u.shape[-1],
                            output_dim=hidden_dim)
    u_transform = input_to_state.apply(u)
    RNN = SimpleRecurrent(activation=Tanh(),
                          dim=hidden_dim, name="RNN")
    h = RNN.apply(u_transform)  # h is hidden states in the RNN
    state_to_output = Linear(name='state_to_output',
                             input_dim=hidden_dim,
                             output_dim=num_classes + 1)
    y_hat = tensor.nnet.softmax(
        state_to_output.apply(h).reshape((-1, num_classes + 1))
    ).reshape((h.shape[0], h.shape[1], -1))
    #T x B x C --> T x C x B
    y_hat = y_hat.dimshuffle(0, 2, 1)
    y_hat.name = 'output_sequences'

    predict = theano.function(inputs=[u, ], outputs=y_hat)

    # Cost
    y = tensor.matrix('target_sequences', dtype='int32')
    #y = y.dimshuffle(0, 1, 2, 'x')
    # Note that y and y_hat don't have the same dimensions.
    # And dimension of y_hat is equal to mask_inputs
    cost = CTC().apply(y, y_hat, l_mask, i_mask)
    #cost = SquaredError().apply(y, y_hat)
    cost.name = 'CTC'

    # Initialization
    for brick in (RNN, state_to_output, input_to_state):
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0)
        brick.initialize()

    # Data
    dataset = IterableDataset({'input_sequences': seq_u,
                               'mask_inputs': mask_inputs,
                               'target_sequences': seq_y,
                               'mask_labels': mask_labels})
    stream = DataStream(dataset)

    # Training
    algorithm = GradientDescent(cost=cost,
                                params=ComputationGraph(cost).parameters,
                                step_rule=CompositeRule([StepClipping(10.0),
                                                         Scale(0.01)]))
    monitor_cost = TrainingDataMonitoring([cost],
                                          prefix="train",
                                          after_every_epoch=True)

    y_hat_shape = y_hat.shape
    y_hat_shape.name = 'Shape'
    monitor_output = TrainingDataMonitoring([y_hat_shape],
                                            prefix="y_hat",
                                            every_n_epochs=2)

    model = Model(cost)
    main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                         extensions=[monitor_cost, monitor_output,
                                     FinishAfter(after_n_epochs=n_epochs),
                                     Printing()],
                         model=model)

    main_loop.run()


if __name__ == "__main__":
    nClasses = 4
    logging.basicConfig(level=logging.INFO)
    with open(sys.argv[1], "rb") as pkl_file:
        data = pickle.load(pkl_file)
    # S x T x B x D --> S x T x D x B
    inputs = data['inputs'].transpose(0, 1, 3, 2)
    labels = data['labels']
    mask_inputs = data['mask_inputs'].transpose(0, 1, 3, 2)
    mask_inputs = numpy.ones((mask_inputs.shape[0], mask_inputs.shape[1], mask_inputs.shape[3]))
    mask_labels = data['mask_labels']
    for i in range(len(labels)):
        batch = []
        for col in labels[i].T:
            y1 = [nClasses]
            for char in col:
                y1 += [char, nClasses]
            batch.append(y1)
        labels[i] = numpy.asarray(batch).transpose(1, 0)
    print mask_inputs.shape

    main(inputs, labels, mask_inputs, mask_labels, 9, nClasses, 10)
