from deuces import Card
from rlofc.royalty_calculator import RoyaltyCalculator
from rlofc.ofc_evaluator import OFCEvaluator


evaluator = OFCEvaluator()


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

    def get_rank(self):
        return evaluator.evaluate(self.cards, [])


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

    def get_free_streets(self):
        """Return a binary list of available streets, FMB."""
        available = [
            1 if self.front.length() < 3 else 0,
            1 if self.mid.length() < 5 else 0,
            1 if self.back.length() < 5 else 0
        ]

        return available



    def is_complete(self):
        if self.back.length() == 5 and \
                self.mid.length() == 5 and \
                self.front.length() == 3:
            return True
        return False

    def is_foul(self):
        if not self.is_complete():
            raise ValueError("Board is incomplete!")

        if self.front.get_rank() >= \
                self.mid.get_rank() >= \
                self.back.get_rank():
            return False

        return True
