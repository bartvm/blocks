import numpy as np
import theano
import theano.tensor as T
import logging
from blocks.bricks import Linear, Tanh, Softmax
from blocks.bricks.cost import SquaredError
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.bricks import WEIGHTS
from blocks.initialization import IsotropicGaussian, Constant
from blocks.datasets import Dataset, ContainerDataset
from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.algorithms import (GradientDescent, Scale, 
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import SimpleRecurrent
import matplotlib.pyplot as plt


class ToyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def get_data(self, state=None, request=None):
        return self.data[request]


def main(seq_u, seq_y, n_h, n_y, n_epochs):
    # Building Model
    u = T.tensor3('input_sequence')
    input_to_state = Linear(name='input_to_state',
                            input_dim=seq_u.shape[-1],
                            output_dim=n_h)
    u_transform = input_to_state.apply(u)
    RNN = SimpleRecurrent(activation=Tanh(),
                          dim=n_h, name="RNN")
    h = RNN.apply(u_transform)  # h is hidden states in the RNN
    state_to_output = Linear(name='state_to_output',
                             input_dim=n_h,
                             output_dim=seq_y.shape[-1])
    y_hat = state_to_output.apply(h)
    y_hat.name = 'output_sequence'

    predict = theano.function(inputs = [u, ], outputs = y_hat)

    # Cost
    y = T.tensor3('target_sequence')
    cost = SquaredError().apply(y, y_hat)
    cost.name = 'MSE'

    # Initialization
    RNN.weights_init = state_to_output.weights_init = \
        input_to_state.weights_init = IsotropicGaussian(0.01)
    RNN.biases_init = state_to_output.biases_init = \
        input_to_state.biases_init = Constant(0)
    RNN.initialize()
    state_to_output.initialize()
    input_to_state.initialize()

    # Data
    dataset = ContainerDataset({'input_sequence': seq_u, 'target_sequence': seq_y})
    stream = DataStream(dataset)

    # Training
    algorithm = GradientDescent(cost=cost, 
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
    plt.close('all')
    fig = plt.figure()

    # Graph 1
    ax1 = plt.subplot(211)
    plt.plot(test_u[:, 0, :])
    plt.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = plt.subplot(212)
    true_targets = plt.plot(test_y[:, 0, :])

    guessed_targets = plt.plot(test_y_hat[:, 0, :], linestyle='--')
    plt.grid()
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    ax1.annotate('Input data point', xy=(2, test_u[2, 0, 0]), 
                 xytext=(2, test_u[2, 0, 0] + 1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.annotate('Output data point (Same point but with 2 time_steps delay)', 
                 xy=(4, test_y[4, 0, 0]), xytext=(4, test_y[4, 0, 0] + 1),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Save as a file
    plt.savefig('RNN_seq.png')
    print 'Figure is saved as a .png file.'

    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n_examples = 20
    batch_size = 10
    n_u = 2 # input vector size
    n_h = 7 # hidden vector size
    n_y = 2 # output vector size
    time_steps = 15 # number of time-steps in time
    n_seq = 10 # number of sequences for training

    np.random.seed(0)
    
    # generating random sequences
    seq_u = np.random.randn(n_examples,  time_steps, batch_size, n_u)
    seq_y = np.zeros((n_examples,  time_steps, batch_size, n_y))

    seq_y[:, 2:, :, 0] = seq_u[:, :-2, :, 0] # 2 time-step delay between input and output
    seq_y[:, 4:, :, 1] = seq_u[:, :-4, :, 1] # 4 time-step delay

    seq_y += 0.01 * np.random.standard_normal(seq_y.shape)

    main(seq_u, seq_y, 8, 2, 1000)


