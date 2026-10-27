from .update_rule import UpdateRule


class DelayFilteredSGD(UpdateRule):
    """Reject arrivals older than an inclusive configurable model-age limit."""
    def __init__(self, threshold):
        super().__init__(threshold=threshold)
