import logging
import signal
import time
from subprocess import Popen, PIPE

try:
    from bokeh.plotting import figure, output_server, push, show, curdoc
    from bokeh.session import Session
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from blocks import config
from blocks.extensions import SimpleExtension

logger = logging.getLogger(__name__)


class Plot(SimpleExtension):
    """Live plotting of monitoring channels.

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
    server_url : str, optional
        Url of the bokeh-server. Ex: when starting the bokeh-server with
        ``bokeh-server --ip 0.0.0.0`` at ``alice``, server_url should be
        ``http://alice:5006``. When not specified the default configured
        by ``bokeh_server`` in ``.blocksrc`` will be used. Defaults to
        ``http://localhost:5006/``.

    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, channels, open_browser=False,
                 start_server=False, server_url=None, **kwargs):
        if not BOKEH_AVAILABLE:
            raise ImportError

        if server_url is None:
            server_url = config.bokeh_server

        self.data_sources = {}
        self.start_server = start_server
        self.document = document
        self.server_url = server_url

        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault("before_first_epoch", True)

        # set x_axis_label to the most frequent option
        self.x_axis_label = ''
        if (kwargs.get('after_epoch') or kwargs.get('after_n_epochs') or
                kwargs.get('every_n_epochs')):
            self.x_axis_label = 'epochs'
        if (kwargs.get('after_batch') or kwargs.get('after_n_batches') or
                kwargs.get('every_n_batches')):
            self.x_axis_label = 'batches'

        self._startserver()

        # Create figures for each group of channels
        self.p = []
        self.p_indices = {}
        for i, channel_set in enumerate(channels):
            fig = figure(title='{} #{}'.format(self.document, i + 1),
                         x_axis_label=self.x_axis_label,
                         y_axis_label='value')
            fig.axis.major_label_standoff = 1

            self.p.append(fig)
            for channel in channel_set:
                self.p_indices[channel] = i
        if open_browser:
            show()
        else:
            push()

        super(Plot, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        if self.x_axis_label == 'epochs':
            x = log.status['epochs_done']
        elif self.x_axis_label == 'batches':
            x = log.status['iterations_done']
        else:
            raise ValueError('x_axis_label should be either epoch or batches')

        i = 0
        for key in self.p_indices.keys():
            if key in log.current_row:
                value = log.current_row[key]
                if key not in self.data_sources:
                    fig = self.p[self.p_indices[key]]
                    fig.line(x=[x], y=[value], legend=key,
                             name=key,
                             line_color=self.colors[i % len(self.colors)])
                    i += 1
                    renderer = fig.select(dict(name=key))
                    self.data_sources[key] = renderer[0].data_source
                    push()
                else:
                    self.data_sources[key].data['x'].append(x)
                    self.data_sources[key].data['y'].append(value)

                    self.session.store_objects(self.data_sources[key])

    def _startserver(self):
        if self.start_server:
            def preexec_fn():
                """Prevent the server from dying on training interrupt."""
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

        self.session = Session(
            root_url=self.server_url,
            load_from_config=False)
        output_server(self.document, session=self.session)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._startserver()
        curdoc().add(*self.p)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sub'] = None
        return state


class PlotHistogram(SimpleExtension):
    def __init__(self, document, channels, server_url='default', **kwargs):
        """Live plot an histogram of the mean of the provided channels.

        This extension live plots an histogram of the mean value of each of
        the provided channels over the monitored dataset. It will also
        display confidence intervals representing the standard deviation of
        the channel over the dataset.

        .. warning::
            This extension assumes that a bokeh server is running at the
            provided url and won't try to start a server autonomously. To
            start a bokeh server: ``bokeh-server --ip=<ip> --port=<port>``.
            If you want the server to accept connections from any address
            use ``<ip>=0.0.0.0``.

        Parameters
        ----------
        document : string
            The name of the document.
        channels : list
            The channels to be displayed as a histogram.
        server_url : string
            The url of the server.

        """
        self.document = document
        self.channels = channels
        self.server_url = server_url

        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault("before_first_epoch", False)
        super(PlotHistogram, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        channels = self.channels
        log = self.main_loop.log
        epoch = log.status['epochs_done']

        # create a new document --> a new session
        self.session = Session(
            root_url=self.server_url,
            load_from_config=False)

        output_server(self.document+' (epoch'+str(epoch)+')',
                      session=self.session, clear=True)

        for key in channels:
            if key in log.current_row:
                value = log.current_row[key]
                kl_mean = value.mean(axis=0)
                kl_std_dev = value.std(axis=0)
                nhid = len(kl_mean)

                fig = figure(title=key, plot_width=1000,
                             x_axis_label='epoch',
                             y_axis_label=self.document)
                # display alternating color histograms
                fig.quad(top=kl_mean[::2], bottom=0, left=range(nhid)[::2],
                         right=range(nhid+1)[1::2], fill_color='#66E066',
                         line_color="black")
                fig.quad(top=kl_mean[1::2], bottom=0, left=range(nhid)[1::2],
                         right=range(nhid+1)[2::2], fill_color='#1B72C3',
                         line_color="black")

                # display confidence intervals (std_dev)
                # intervals' max height to 20% of the figure
                kl_std_dev = kl_std_dev.astype(float) / max(
                    kl_std_dev).astype(float)
                kl_std_dev = kl_std_dev * max(kl_mean)/5.0
                # position relatively to the previous histogram
                std_hist_bottom = kl_mean - kl_std_dev/2.0
                std_hist_top = kl_mean + kl_std_dev/2.0
                fig.quad(top=std_hist_top, bottom=std_hist_bottom,
                         left=[el + 0.47 for el in range(nhid)],
                         right=[el + 0.53 for el in range(nhid)],
                         fill_color='black', line_color="black")
        push()
