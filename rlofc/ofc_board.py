from deuces import Card, Evaluator


evaluator = Evaluator()


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

    def get_royalties(self):
        if len(self.cards) == 5:
            return self.get_royalties_5()
        if len(self.cards) == 3:
            return self.get_royalties_3()

    def get_royalties_5(self):
        rank = evaluator.evaluate([], self.cards)

        # I would love to do this with evaluator.get_rank_class,
        # but it's broken: https://github.com/worldveil/deuces/issues/9
        if rank > 1609:
            return 0   # Nothing good enough

        if rank > 1599:
            return 2   # Straight

        if rank > 322:
            return 4   # Flush

        if rank > 166:
            return 6   # Full house

        if rank > 10:
            return 10  # Four-of-a-kind

        if rank > 1:
            return 15  # Straight flush

        if rank == 1:
            return 35  # Royal flush


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

