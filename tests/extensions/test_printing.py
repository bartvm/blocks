import sys
from StringIO import StringIO

from blocks.extensions import FinishAfter, Printing
from blocks.utils.testing import MockMainLoop


def setup_mainloop(print_status):
    epochs = 1
    main_loop = MockMainLoop(extensions=[Printing(print_status),
                                         FinishAfter(after_n_epochs=epochs)])
    return main_loop

def test_printing_status():
    main_loop = setup_mainloop(print_status=True)

    stdout = sys.stdout
    try:
        saved_out = StringIO()
        sys.stdout = saved_out

        main_loop.run()
        output = saved_out.getvalue()
        assert 'Training status' in output
    finally:
        sys.stdout = stdout

def test_printing_no_status():
    main_loop = setup_mainloop(print_status=False)

    stdout = sys.stdout
    try:
        saved_out = StringIO()
        sys.stdout = saved_out

        main_loop.run()
        output = saved_out.getvalue()
        assert 'Training status' not in output
    finally:
        sys.stdout = stdout
