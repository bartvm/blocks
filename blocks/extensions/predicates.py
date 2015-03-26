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
        return log.current_entry.get(self.record_name, False)
