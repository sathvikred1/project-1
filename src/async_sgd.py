from .update_rule import UpdateRule


class AsyncSGD(UpdateRule):
    """Apply each stored gradient immediately, irrespective of staleness."""
    def __init__(self):
        super().__init__()
