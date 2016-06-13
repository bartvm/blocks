"""Module level configuration.

Blocks allows module-wide configuration values to be set using a YAML_
configuration file and `environment variables`_. Environment variables
override the configuration file which in its turn overrides the defaults.

The configuration is read from ``~/.blocksrc`` if it exists. A custom
configuration file can be used by setting the ``BLOCKS_CONFIG`` environment
variable. A configuration file is of the form:

.. code-block:: yaml

   data_path: /home/user/datasets

If a setting is not configured and does not provide a default, a
:class:`~.ConfigurationError` is raised when it is
accessed.

Configuration values can be accessed as attributes of
:const:`blocks.config.config`.

    >>> from blocks.config import config
    >>> print(config.default_seed) # doctest: +SKIP
    1

The following configurations are supported:

.. option:: default_seed

   The seed used when initializing random number generators (RNGs) such as
   NumPy :class:`~numpy.random.RandomState` objects as well as Theano's
   :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams` objects. Must be an
   integer. By default this is set to 1.

.. option:: recursion_limit

   The recursion max depth limit used in
   :class:`~blocks.main_loop.MainLoop` as well as in other situations when
   deep recursion is required. The most notable example of such a situation
   is pickling or unpickling a complex structure with lots of objects, such
   as a big Theano computation graph.

.. option:: profile, BLOCKS_PROFILE

   A boolean value which determines whether to print profiling information
   at the end of a call to :meth:`.MainLoop.run`.

.. option:: log_backend

   The backend to use for logging experiments. Defaults to `python`, which
   stores the log as a Python object in memory. The other option is
   `sqlite`.

.. option:: sqlite_database, BLOCKS_SQLITEDB

   The SQLite database file to use.

.. option:: max_blob_size

   The maximum size of an object to store in an SQLite database in bytes.
   Objects beyond this size will trigger a warning. Defaults to 4 kilobyte.

.. option:: temp_dir

   The directory in which Blocks will create temporary files. If
   unspecified, the platform-dependent default chosen by the Python
   ``tempfile`` module is used.

.. _YAML: http://yaml.org/
.. _environment variables:
   https://en.wikipedia.org/wiki/Environment_variable

"""
import logging
import os

import six
import yaml

logger = logging.getLogger(__name__)

NOT_SET = object()


class ConfigurationError(Exception):
    """Error raised when a configuration value is requested but not set."""
    pass


class Configuration(object):
    def __init__(self):
        self.config = {}

    def load_yaml(self):
        if 'BLOCKS_CONFIG' in os.environ:
            yaml_file = os.environ['BLOCKS_CONFIG']
        else:
            yaml_file = os.path.expanduser('~/.blocksrc')
        if os.path.isfile(yaml_file) and os.path.getsize(yaml_file):
            with open(yaml_file) as f:
                config_dict = yaml.safe_load(f) or {}
                for key, value in config_dict.items():
                    if key not in self.config:
                        raise ValueError("Unrecognized config in YAML: {}"
                                         .format(key))
                    self.config[key]['yaml'] = value

    def __getattr__(self, key):
        if key == 'config' or key not in self.config:
            raise AttributeError
        config_setting = self.config[key]
        if 'value' in config_setting:
            value = config_setting['value']
        elif ('env_var' in config_setting and
              config_setting['env_var'] in os.environ):
            value = os.environ[config_setting['env_var']]
        elif 'yaml' in config_setting:
            value = config_setting['yaml']
        elif 'default' in config_setting:
            value = config_setting['default']
        else:
            raise ConfigurationError("Configuration not set and no default "
                                     "provided: {}.".format(key))
        return config_setting['type'](value)

    def __setattr__(self, key, value):
        if key != 'config' and key in self.config:
            self.config[key]['value'] = value
        else:
            super(Configuration, self).__setattr__(key, value)

    def add_config(self, key, type_, default=NOT_SET, env_var=None):
        """Add a configuration setting.

        Parameters
        ----------
        key : str
            The name of the configuration setting. This must be a valid
            Python attribute name i.e. alphanumeric with underscores.
        type : function
            A function such as ``float``, ``int`` or ``str`` which takes
            the configuration value and returns an object of the correct
            type.  Note that the values retrieved from environment
            variables are always strings, while those retrieved from the
            YAML file might already be parsed. Hence, the function provided
            here must accept both types of input.
        default : object, optional
            The default configuration to return if not set. By default none
            is set and an error is raised instead.
        env_var : str, optional
            The environment variable name that holds this configuration
            value. If not given, this configuration can only be set in the
            YAML configuration file.

        """
        self.config[key] = {'type': type_}
        if env_var is not None:
            self.config[key]['env_var'] = env_var
        if default is not NOT_SET:
            self.config[key]['default'] = default


def bool_(val):
    """Like `bool`, but the string 'False' evaluates to `False`."""
    if isinstance(val, six.string_types) and val.lower() == 'false':
        return False
    return bool(val)


def str_or_none(val):
    """Like str, but allows a value of None to pass through as-is."""
    return str(val) if val is not None else None


# Define configuration options
config = Configuration()
config.add_config('default_seed', type_=int, default=1)
config.add_config('recursion_limit', type_=int, default=10000)
config.add_config('profile', type_=bool_, default=False,
                  env_var='BLOCKS_PROFILE')
config.add_config('log_backend', type_=str, default='python')
config.add_config('sqlite_database', type_=str,
                  default=os.path.expanduser('~/blocks_log.sqlite'),
                  env_var='BLOCKS_SQLITEDB')
config.add_config('max_blob_size', type_=int, default=4096)
config.add_config('temp_dir', type_=str_or_none, default=None)
config.load_yaml()
