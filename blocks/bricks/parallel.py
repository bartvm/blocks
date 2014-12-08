"""Generic transformations with multiple inputs and/or outputs.

As bricks from this module are intended to serve as adapters, most of them
are lazy-only, i.e. can not be initialized with a single constructor call.

"""
import copy

from blocks.bricks import Brick, lazy, application, MLP, Identity, Initializeable
from blocks.utils import update_instance


class Parallel(Brick, Initializeable):
    """Apply similar transformations to several channels.

    Parameters
    ----------
    channel_names : list of str
        Input names.
    input_dims : dict
        Dictonary of input dimensions, keys are channel names, values are
        dimensions.
    output_dims : dict
        Dictonary of output dimensions, keys are channel names, values are
        dimensions.
    prototype : :class:`Brick`
        A transformation prototype. A copy will be created for every channel.
        If ``None``, a linear transformation is used.

    """
    @lazy
    def __init__(self, channel_names, input_dims, output_dims,
                 prototype=None, weights_init=None, biases_init=None,
                 **kwargs):
        super(Parallel, self).__init__(**kwargs)
        update_instance(self, locals())

        if not self.prototype:
            self.prototype = MLP([Identity()], use_bias=False)
        self.transforms = []
        for name in self.channel_names:
            self.transforms.append(copy.deepcopy(self.prototype))
            self.transforms[-1].name = "transform_{}".format(name)
        self.children = self.transforms
        self.push_init_bias = True

    def _push_allocation_config(self):
        for name, transform in zip(self.channel_names, self.transforms):
            transform.dims[0] = self.input_dims[name]
            transform.dims[-1] = self.output_dims[name]

    @application
    def apply(self, **kwargs):
        return [transform.apply(kwargs[name])
                for name, transform
                in zip(self.channel_names, self.transforms)]

    @apply.property('inputs')
    def apply_inputs(self):
        return self.channel_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.channel_names


class Fork(Parallel):
    """Forks single input into multiple channels.

    Parameters
    ----------
    fork_names : list of str
        Names of the channels to fork.
    prototype : instance of :class:`Brick`
        A prototype for the input-to-fork transformations. A copy will be
        created for every output channel.

    Attributes
    ----------
    input_dim : int
        The input dimension. Required for allocation.
    output_dims : dict of (output_name, int) pairs
        The dimesions of the outputs. Required for allocation.

    Notes
    -----
        Lazy initialization only.

    """
    def __init__(self, fork_names, prototype=None, **kwargs):
        super(Fork, self).__init__(fork_names, prototype=prototype, **kwargs)
        self.fork_names = fork_names

    def _push_allocation_config(self):
        self.input_dims = {name: self.input_dim for name in self.fork_names}
        self.output_dims = self.fork_dims
        super(Fork, self)._push_allocation_config()

    @application(inputs='inp')
    def apply(self, inp):
        return super(Fork, self).apply(**{name: inp
                                          for name in self.fork_names})

    @apply.property('outputs')
    def apply_outputs(self):
        return super(Fork, self).apply.outputs


class Mixer(Parallel):
    """Mixes a new channel with old ones.

    .. digraph:: mixer

       rankdir=TB;
       splines=line;
       subgraph cluster_0 {
         label="old_inputs";
         a[label=""]; b[label=""]; c[label=""];
       }
       subgraph cluster_1 {
         label="outputs"; labelloc=b;
         d[label=""]; e[label=""]; f[label=""];
       }
       a -> d;
       b -> e;
       c -> f;
       new_input -> d;
       new_input -> e;
       new_input -> f;

    Parameters
    ----------
    old_names : list of str
        The names of the old channels.
    new_name : list of str
        The new channel's name.

    Attributes
    ----------
    channel_dims : dict of (channel_name, int) pairs
        The dimensions of the channels. Required for allocation.

    """
    def __init__(self, old_names, new_name, prototype=None, **kwargs):
        super(Mixer, self).__init__(old_names, prototype=prototype, **kwargs)
        self.old_names = old_names
        self.new_name = new_name

    def _push_allocation_config(self):
        self.input_dims = {name: self.channel_dims[self.new_name]
                           for name in self.old_names}
        self.output_dims = {name: self.channel_dims[name]
                            for name in self.old_names}
        super(Mixer, self)._push_allocation_config()

    @application
    def apply(self, **kwargs):
        new = kwargs.pop(self.new_name)
        assert set(kwargs.keys()) == set(self.old_names)
        result = super(Mixer, self).apply(
            return_list=True, **{name: new for name in self.old_names})
        for i, name in enumerate(self.old_names):
            result[i] += kwargs[name]
        return result

    @apply.property('inputs')
    def apply_inputs(self):
        return [self.new_name] + self.old_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.old_names
