from .update_rule import UpdateRule


class SynchronousSGD(UpdateRule):
    """One gradient per worker, with a barrier before every averaged update."""
    synchronous = True

    def __init__(self, workers):
        super().__init__(gradients_per_update=workers)
