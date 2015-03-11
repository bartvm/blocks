import logging
import signal
import time
from subprocess import Popen, PIPE

import numpy
try:
    from bokeh.plotting import (curdoc, cursession, figure, output_server,
                                push, show)
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
from blocks.extensions import SimpleExtension

logger = logging.getLogger(__name__)


class PlotManager(SimpleExtension):
    """Live plotting.

    In most cases it is preferable to start the Bokeh plotting server
    manually, so that your plots are stored permanently.

    Alternatively, you can set the ``start_server`` argument of this
    extension to ``True``, to automatically start a server when training
    starts. However, in that case your plots will be deleted when you shut
    down the plotting server!

    .. warning::

       When starting the server automatically using the ``start_server``
       argument, the extension won't attempt to shut down the server at the
       end of training (to make sure that you do not lose your plots the
       moment training completes). You have to shut it down manually (the
       PID will be shown in the logs). If you don't do this, this extension
       will crash when you try and train another model with
       ``start_server`` set to ``True``, because it can't run two servers
       at the same time.

    Parameters
    ----------
    document : str
        The name of the Bokeh document. Use a different name for each
        experiment if you are storing your plots.
    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    open_browser : bool, optional
        Whether to try and open the plotting server in a browser window.
        Defaults to ``True``. Should probably be set to ``False`` when
        running experiments non-locally (e.g. on a cluster or through SSH).
    start_server : bool, optional
        Whether to try and start the Bokeh plotting server. Defaults to
        ``False``. The server started is not persistent i.e. after shutting
        it down you will lose your plots. If you want to store your plots,
        start the server manually using the ``bokeh-server`` command. Also
        see the warning above.

    """
    def __init__(self, document, plotters, open_browser=False,
                 start_server=False, **kwargs):
        if not BOKEH_AVAILABLE:
            raise ImportError
        self.plotters = plotters
        self.start_server = start_server
        self.document = document
        self._startserver()

        for plotter in self.plotters:
            plotter.manager = self
            plotter.initialize()
        if open_browser:
            show()

        kwargs.setdefault('after_every_epoch', True)
        kwargs.setdefault("before_first_epoch", True)
        super(PlotManager, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        for plotter in self.plotters:
            plotter.call()

        push()

    def _startserver(self):
        if self.start_server:
            def preexec_fn():
                """Prevents the server from dying on training interrupt."""
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Only memory works with subprocess, need to wait for it to start
            logger.info('Starting plotting server on localhost:5006')
            self.sub = Popen('bokeh-server --ip 0.0.0.0 '
                             '--backend memory'.split(),
                             stdout=PIPE, stderr=PIPE, preexec_fn=preexec_fn)
            time.sleep(2)
            logger.info('Plotting server PID: {}'.format(self.sub.pid))
        else:
            self.sub = None
        output_server(self.document)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sub'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._startserver()
        curdoc().add(*self.p)


class BasePlotter(object):
    """Base class for plotter."""
    @property
    def manager(self):
        return self._manager

    @manager.setter
    def manager(self, manager):
        self._manager = manager

    def initialize(self):
        """Initialize the plots state."""
        pass

    def call(self):
        """Update the plots"""
        pass

    def set_titles(self, titles, matching_list):
        if isinstance(titles, str):
            self.titles = [titles] * len(matching_list)
        else:
            if not isinstance(titles, list):
                raise ValueError('titles argument must be a '
                                 'list of strings or a string.')
            if len(titles) == 1:
                self.titles = titles * len(matching_list)
            elif len(titles) != len(matching_list):
                raise ValueError('titles must have same size as '
                                 'the number of figures.')
            else:
                self.titles = titles


class Plotter(BasePlotter):
    """Plotting of monitoring channels.

    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    titles : list of strings or string
        The name of the associated plots. If t

    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, channels, titles=''):
        self.channels = channels
        self.set_titles(titles, channels)

        super(Plotter, self).__init__()

    def initialize(self):
        channels = self.channels
        titles = self.titles
        self.plots = {}
        self.p = []
        self.p_indices = {}
        for i, (channel_set, title) in enumerate(zip(channels, titles)):
            self.p.append(figure(title=title))
            for channel in channel_set:
                self.p_indices[channel] = i

    def call(self):
        log = self.manager.main_loop.log
        iteration = log.status.iterations_done
        i = 0
        for key, value in log.current_row:
            if key in self.p_indices:
                if key not in self.plots:
                    fig = self.p[self.p_indices[key]]
                    fig.line([iteration], [value], legend=key,
                             x_axis_label='iterations',
                             y_axis_label='value', name=key,
                             line_color=self.colors[i % len(self.colors)])
                    i += 1
                    renderer = fig.select(dict(name=key))
                    self.plots[key] = renderer[0].data_source
                else:
                    self.plots[key].data['x'].append(iteration)
                    self.plots[key].data['y'].append(value)

                    cursession().store_objects(self.plots[key])


class DisplayImage(BasePlotter):
    """Show images.

    Parameters
    ----------
    image_getters : list of callable objects
        A list of function without argument returning images to display.
        The returned images must be in (N, M, 3), (N, M, 1) or (N, M) and
        must be between 0 and 255.
    """

    def __init__(self, image_getters, titles=''):
        self.image_getters = image_getters
        self.set_titles(titles, image_getters)

        super(DisplayImage, self).__init__()

    def initialize(self):
        self.figs = []
        i = 0
        for image_getter, title in zip(self.image_getters, self.titles):
            self.figs.append(figure(title=title,
                                    x_range=[0, 1],
                                    y_range=[0, 1]))
            i += 1

    def call(self):
        for fig, image_getter in zip(self.figs, self.image_getters):
            rgba_image = format_image_to_rgba(image_getter()).copy()
            img_h = rgba_image.shape[0]
            img_w = rgba_image.shape[1]
            fig.renderers = []

            fig.image_rgba(image=[rgba_image],
                           x=[0], y=[0],
                           dw=[min(img_w * 1. / img_h, 1)],
                           dh=[min(img_h * 1. / img_w, 1)])


def format_image_to_rgba(image_0255):
    image_0255_clipped = numpy.clip(image_0255, 0, 255)
    int_image = numpy.round(
        image_0255_clipped
    ).astype(dtype=numpy.uint8)

    rgba_image = numpy.zeros(int_image.shape[:2], dtype=numpy.uint32)
    image_view = rgba_image.view(
        dtype=numpy.uint8
    ).reshape(int_image.shape[:2] + (4, ))
    image_view[:, :, 3] = 255

    if int_image.ndim == 2:
        image_view[:, :, :3] = int_image[:, :, None]
    elif int_image.ndim == 3:
        image_view[:, :, :3] = int_image
    else:
        image_view = int_image

    return rgba_image
