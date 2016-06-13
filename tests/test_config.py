import os
import tempfile

from numpy.testing import assert_raises

from blocks.config import Configuration, ConfigurationError


def load_config(contents):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(contents)
        filename = f.name
    os.environ['BLOCKS_CONFIG'] = filename
    if 'BLOCKS_DATA_PATH' in os.environ:
        del os.environ['BLOCKS_DATA_PATH']
    config = Configuration()
    config.add_config('data_path', str, env_var='BLOCKS_DATA_PATH')
    config.add_config('config_with_default', int, default='1',
                      env_var='BLOCKS_CONFIG_TEST')
    config.add_config('config_without_default', str)
    config.load_yaml()
    return config


class TestConfig(object):
    def setUp(self):
        self._environ = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._environ)

    def test_config(self):
        config = load_config('data_path: yaml_path')
        assert config.data_path == 'yaml_path'
        os.environ['BLOCKS_DATA_PATH'] = 'env_path'
        assert config.data_path == 'env_path'
        assert config.config_with_default == 1
        os.environ['BLOCKS_CONFIG_TEST'] = '2'
        assert config.config_with_default == 2
        assert_raises(AttributeError, getattr, config,
                      'non_existing_config')
        assert_raises(ConfigurationError, getattr, config,
                      'config_without_default')
        config.data_path = 'manual_path'
        assert config.data_path == 'manual_path'
        config.new_config = 'new_config'
        assert config.new_config == 'new_config'

    def test_empty_config(self):
        load_config('')
