"""Small update policies; timing and worker scheduling live in simulation.py."""
import numpy as np


class UpdateRule:
    synchronous = False

    def __init__(self, gradients_per_update=1, threshold=None):
        self.gradients_per_update = gradients_per_update
        self.threshold = threshold
        self.buffer = []

    def accepts(self, staleness):
        # Equality is accepted, as specified in the project brief.
        return self.threshold is None or staleness <= self.threshold

    def add(self, gradient):
        self.buffer.append(gradient)
        if len(self.buffer) < self.gradients_per_update:
            return None
        # The server model stays fixed while this buffer fills, so accepted
        # gradients cannot become older before this particular batch commits.
        averaged = np.mean(self.buffer, axis=0)
        self.buffer.clear()
        return averaged
