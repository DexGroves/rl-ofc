from treys import Card
from rlofc.rlofc.royalty_calculator import RoyaltyCalculator
from rlofc.rlofc.ofc_evaluator import OFCEvaluator


evaluator = OFCEvaluator()


class OFCHand(object):
    """An OFC street (back, mid, or front).
    Accepts a list of cards and new cards as strings.
    """

    def __init__(self, card_strs):
        self.cards = [Card.new(x) for x in card_strs]

    def add_card(self, new_card_str):
        # if len(new_card_str)==1:
        self.cards.append(Card.new(new_card_str))

        # else:
        #     for i in new_card_str:
        #         self.cards.append(Card.new(i))

    def length(self):
        return len(self.cards)

    def get_rank(self):
        return evaluator.evaluate(self.cards, [])


class OFCBoard(object):
    """Represent the three streets of an OFC game for one player."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.front = OFCHand([])
        self.mid = OFCHand([])
        self.back = OFCHand([])

    def pretty(self):
        print('Front:')
        Card.print_pretty_cards(self.front.cards)
        print('Mid:')
        Card.print_pretty_cards(self.mid.cards)
        print('Back:')
        Card.print_pretty_cards(self.back.cards)

    def get_royalties(self):
        if not self.is_complete():
            return 0

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

    def get_free_street_indices(self):
        available = []
        if self.front.length() < 3:
            available.append(0)
        if self.mid.length() < 5:
            available.append(1)
        if self.back.length() < 5:
            available.append(2)
        return available

    def place_card_by_id(self, card, street_id):
        if street_id == 0:
            self.front.add_card(card)
        if street_id == 1:
            self.mid.add_card(card)
        if street_id == 2:
            self.back.add_card(card)

    def is_complete(self):
        if self.back.length() == 5 and \
                self.mid.length() == 5 and \
                self.front.length() == 3:
            return True
        return False

    def is_foul(self):
        if not self.is_complete():
            return True

        if self.front.get_rank() >= \
                self.mid.get_rank() >= \
                self.back.get_rank():
            return False

        return True
    
    def is_fantasy(self):
        if self.is_foul():
            return False
        
        if self.front.get_rank() >= \
                self.mid.get_rank() >= \
                self.back.get_rank():
            return True
        
        return False

