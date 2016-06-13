from mock import Mock
from numpy.testing import assert_raises

from blocks.extensions import SimpleExtension, CompositeExtension
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.predicates import OnLogRecord


def test_parse_args():
    assert (SimpleExtension.parse_args('before_batch', ('a', 'b')) ==
            (('a',), ('b',)))
    assert (SimpleExtension.parse_args('before_epoch', ('a', 'b')) ==
            ((), ('a', 'b')))


def test_add_list_condition():
    extension_list = Checkpoint('extension_list').add_condition(
        ['before_first_epoch', 'after_epoch'],
        OnLogRecord('notification_name'),
        ('dest_path.kl',))
    extension_iter = Checkpoint('extension_iter')
    extension_iter.add_condition(
        ['before_first_epoch'],
        OnLogRecord('notification_name'),
        ('dest_path.kl',))
    extension_iter.add_condition(
        ['after_epoch'],
        OnLogRecord('notification_name'),
        ('dest_path.kl',))
    assert len(extension_list._conditions) == len(extension_iter._conditions)
    assert_raises(ValueError, extension_iter.add_condition,
                  callbacks_names='after_epoch',
                  predicate=OnLogRecord('notification_name'),
                  arguments=('dest_path.kl',))


def test_composite_extension_main_loop_assignment():
    ext1 = Mock()
    ext2 = Mock()

    comp = CompositeExtension([ext1, ext2])
    comp.main_loop = object()

    assert ext1.main_loop == comp.main_loop
    assert ext2.main_loop == comp.main_loop


def test_composite_extension_dispatches():
    ext1 = Mock()
    ext2 = Mock()

    comp = CompositeExtension([ext1, ext2])
    comp.main_loop = object()

    comp.dispatch('before_training')
    ext1.dispatch.assert_called_once_with('before_training')
    ext2.dispatch.assert_called_once_with('before_training')

    comp.dispatch('after_batch', 5)
    ext1.dispatch.assert_called_with('after_batch', 5)
    ext2.dispatch.assert_called_with('after_batch', 5)


def test_composite_extension_run_before():
    class Foo(SimpleExtension):
        def __init__(self, num, **kwargs):
            self.num = num
            super(Foo, self).__init__(**kwargs)

        def do(self, which_callback, *args):
            self.num += 1

    class Bar(CompositeExtension):
        def do(self, which_callback, *args):
            self.num = 0
            for sub in self.sub_extensions:
                self.num += sub.num

    comp = Bar([Foo(1, before_training=True),
                Foo(2, before_training=True)],
               before_training=True)
    comp.main_loop = Mock()
    comp.dispatch('before_training')

    assert comp.num == 3


def test_composite_extension_run_after():
    class Foo(SimpleExtension):
        def __init__(self, num, **kwargs):
            self.num = num
            super(Foo, self).__init__(**kwargs)

        def do(self, which_callback, *args):
            self.num += 1

    class Bar(CompositeExtension):
        def do(self, which_callback, *args):
            self.num = 0
            for sub in self.sub_extensions:
                self.num += sub.num

    comp = Bar([Foo(1, before_training=True),
                Foo(2, before_training=True)],
               before_training=True,
               run_before_children=False)
    comp.main_loop = Mock()
    comp.dispatch('before_training')

    assert comp.num == 5


def test_composite_extension_different_schedules():
    class Foo(SimpleExtension):
        def __init__(self, **kwargs):
            self.do = Mock()
            super(Foo, self).__init__(**kwargs)

        def do(self, *args):
            pass

    a = Foo(after_batch=False, after_training=True)
    b = Foo(after_batch=True)
    comp = CompositeExtension([a, b], before_training=True)
    comp.main_loop = Mock()
    comp.do = Mock()
    comp.dispatch('before_training')
    comp.dispatch('after_batch')
    comp.dispatch('after_training')
    comp.do.assert_called_once_with('before_training')
    a.do.assert_called_once_with('after_training')
    b.do.assert_called_once_with('after_batch')
