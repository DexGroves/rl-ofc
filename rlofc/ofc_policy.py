import numpy as np
from rlofc.ofc_board import OFCBoard


class OFCPolicy(object):
    """An OFC decision maker."""

    def __init__(self):
        self.board = OFCBoard()

    def place_starting_hand(self, cards):
        pass

    def place_new_card(self, card):
        pass


class OFCRandomPolicy(OFCPolicy):
    """Place cards at random!"""

    def place_starting_hand(self, cards):
        for card in cards:
            self.place_new_card(card)

    def place_new_card(self, card):
        roll = np.random.uniform(0, 1, 3) * self.board.get_free_streets()
        street = np.argmax(roll)

        if street == 0:
            self.board.front.add_card(card)

        if street == 1:
            self.board.mid.add_card(card)

        if street == 2:
            self.board.back.add_card(card)
