import numpy as np


class OFCAgent(object):
    """An OFC decision maker."""

    def place_new_card(self, card, board):
        """Return 0, 1, 2 for front, mid, back."""
        pass


class OFCRandomAgent(OFCAgent):
    """Place cards at random!"""

    def place_new_card(self, card, board):
        roll = np.random.uniform(0, 1, 3) * board.get_free_streets()
        street = np.argmax(roll)
        return street


class OFCRLAgent(OFCAgent):
    """Insert neural network here."""
    pass
