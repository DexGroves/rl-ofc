from deuces import Card
from rlofc.royalty_calculator import RoyaltyCalculator


class OFCHand(object):
    """An OFC street (back, mid, or front).
    Accepts a list of cards and new cards as strings.
    """

    def __init__(self, card_strs):
        self.cards = [Card.new(x) for x in card_strs]

    def add_card(self, new_card_str):
        self.cards.append(Card.new(new_card_str))

    def length(self):
        return len(self.cards)


class OFCBoard(object):
    """Represent the three streets of an OFC game for one player."""

    def set_front(self, cards):
        self.front = OFCHand(cards)

    def set_mid(self, cards):
        self.mid = OFCHand(cards)

    def set_back(self, cards):
        self.back = OFCHand(cards)

    def get_royalties(self):
        if not self.is_complete():
            raise ValueError("Board is incomplete!")

        royalty_total = \
            RoyaltyCalculator.score_front_royalties(self.front.cards) + \
            RoyaltyCalculator.score_mid_royalties(self.mid.cards) + \
            RoyaltyCalculator.score_back_royalties(self.back.cards)

        return royalty_total

    def is_complete(self):
        if self.back.length() == 5 and \
                self.mid.length() == 5 and \
                self.front.length() == 3:
            return True
        return False
