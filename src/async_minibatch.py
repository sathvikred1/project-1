from .update_rule import UpdateRule


class AsyncMiniBatchSGD(UpdateRule):
    """Educational fixed-B asynchronous collection, not paper-exact adaptive MB."""
    def __init__(self, gradients_per_update, threshold):
        super().__init__(gradients_per_update, threshold)
