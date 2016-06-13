from deuces import Card, Evaluator, Deck


class OFCBoard(object):
    """Represent the three streets of an OFC game for one player."""

    def set_front(self, cards):
        if len(cards) > 3:
            raise ValueError("Too many cards for front!")
        self.front = cards

    def set_mid(self, cards):
        if len(cards) > 5:
            raise ValueError("Too many cards for mid!")
        self.front = cards

    def set_back(self, cards):
        if len(cards) > 5:
            raise ValueError("Too many cards for back!")
        self.front = cards

    def add_to_front(self, card):
        if len(front) >= 3:
            raise ValueError("Front is too large!")
        self.front.append(card)

    def add_to_mid(self, card):
        if len(mid) >= 5:
            raise ValueError("Mid is too large!")
        self.mid.append(card)

    def add_to_back(self, card):
        if len(back) >= 5:
            raise ValueError("Back is too large!")
        self.back.append(card)
