from deuces import Card, Evaluator


rank_to_royalty = {
    0: 35,   # Royal flush
    1: 15,   # Straight flush
    2: 10,   # Four-of-a-kind
    3: 6,    # Full house
    4: 4,    # Flush
    5: 2     # Straight
}


class OFCHand(object):
    """An OFC street (back, mid, or front).
    Accepts a list of cards and new cards as strings.
    """

    def __init__(self, card_strs, maxlen):
        if len(card_strs) > maxlen:
            raise ValueError("Too many cards!")

        self.maxlen = maxlen
        self.cards = [Card.new(x) for x in card_strs]

    def add_card(self, new_card_str):
        if len(self.cards) > self.maxlen:
            raise ValueError("Too many cards!")

        self.cards.append(Card.new(new_card_str))

    def get_raw_royalties(self):
        pass


class OFCBoard(object):
    """Represent the three streets of an OFC game for one player."""

    def __init__(self):
        self.evaluator = Evaluator()

    def set_front(self, cards):
        self.front = OFCHand(cards, 3)

    def set_mid(self, cards):
        self.mid = OFCHand(mid, 5)

    def set_back(self, cards):
        self.back = OFCHand(back, 5)

    def get_royalties(self):
        if not self.is_complete():
            raise ValueError("Board is incomplete!")
        pass

    def is_complete(self):
        if len(self.back) == 5 and len(self.mid) == 5 and len(self.front) == 3:
            return True
        return False

