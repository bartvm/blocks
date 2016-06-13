class OnLogRecord(object):
    """Trigger a callback when a certain log record is found.

    Parameters
    ----------
    record_name : str
        The record name to check.

    """
    def __init__(self, record_name):
        self.record_name = record_name

    def __call__(self, log):
        return bool(log.current_row.get(self.record_name, False))

    def __eq__(self, other):
        return (type(other) == type(self) and
                other.record_name == self.record_name)
