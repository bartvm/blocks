import numpy
import theano
import logging
from theano import tensor
from blocks.bricks import Linear, Tanh
from blocks.bricks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.graph import ComputationGraph
from matplotlib import pyplot


def main(seq_u, seq_y, hidden_dim, output_dim, n_epochs):
    # Building Model
    u = tensor.tensor3('input_sequence')
    input_to_state = Linear(name='input_to_state',
                            input_dim=seq_u.shape[-1],
                            output_dim=hidden_dim)
    u_transform = input_to_state.apply(u)
    RNN = SimpleRecurrent(activation=Tanh(),
                          dim=hidden_dim, name="RNN")
    h = RNN.apply(u_transform)  # h is hidden states in the RNN
    state_to_output = Linear(name='state_to_output',
                             input_dim=hidden_dim,
                             output_dim=seq_y.shape[-1])
    y_hat = state_to_output.apply(h)
    y_hat.name = 'output_sequence'

    predict = theano.function(inputs=[u, ], outputs=y_hat)

    # Cost
    y = tensor.tensor3('target_sequence')
    cost = SquaredError().apply(y, y_hat)
    cost.name = 'MSE'

    # Initialization
    for brick in (RNN, state_to_output, input_to_state):
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0)
        brick.initialize()

    # Data
    dataset = IterableDataset({'input_sequence': seq_u,
                                'target_sequence': seq_y})
    stream = DataStream(dataset)

    # Training
    algorithm = GradientDescent(cost=cost,
                                params=ComputationGraph(cost).parameters,
                                step_rule=CompositeRule([StepClipping(10.0),
                                                         Scale(0.01)]))
    monitor = TrainingDataMonitoring([cost],
                                     prefix="train",
                                     after_every_epoch=True)
    main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                         extensions=[monitor,
                                     FinishAfter(after_n_epochs=n_epochs),
                                     Printing()])

    main_loop.run()

    # Visualization
    test_u = seq_u[0, :, 0:1, :]
    test_y = seq_y[0, :, 0:1, :]
    test_y_hat = predict(test_u)

    # We just plot one of the sequences
    pyplot.close('all')
    pyplot.figure()

    # Graph 1
    ax1 = pyplot.subplot(211)
    pyplot.plot(test_u[:, 0, :])
    pyplot.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = pyplot.subplot(212)
    true_targets = pyplot.plot(test_y[:, 0, :])

    guessed_targets = pyplot.plot(test_y_hat[:, 0, :], linestyle='--')
    pyplot.grid()
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    ax1.annotate('Input data point', xy=(2, test_u[2, 0, 0]),
                 xytext=(2, test_u[2, 0, 0] + 1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.annotate('Output data point (Same point but with 2 time_steps delay)',
                 xy=(4, test_y[4, 0, 0]), xytext=(4, test_y[4, 0, 0] + 1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    pyplot.savefig('HERE')
    #pyplot.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    num_batches = 20
    batch_size = 10
    input_dim = 2  # input vector size
    hidden_dim = 7  # hidden vector size
    output_dim = 2  # output vector size
    time_steps = 15  # number of time-steps in time

    numpy.random.seed(0)

    # generating random sequences
    seq_u = numpy.random.randn(num_batches,  time_steps, batch_size, input_dim)
    seq_y = numpy.zeros((num_batches,  time_steps, batch_size, output_dim))

    seq_y[:, 2:, :, 0] = seq_u[:, :-2, :, 0]  # 2 time-step delay
    seq_y[:, 4:, :, 1] = seq_u[:, :-4, :, 1]  # 4 time-step delay

    seq_y += 0.01 * numpy.random.standard_normal(seq_y.shape)

    main(seq_u, seq_y, 8, 2, 100)
