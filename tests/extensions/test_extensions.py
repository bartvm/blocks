from blocks.extensions import SimpleExtension


def test_parse_args():
    assert (SimpleExtension.parse_args('before_batch', ('a', 'b')) ==
            (('a',), ('b',)))
    assert (SimpleExtension.parse_args('before_epoch', ('a', 'b')) ==
            ((), ('a', 'b')))
