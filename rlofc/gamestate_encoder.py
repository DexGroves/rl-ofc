class GamestateBinaryEncoder(object):
    """Encode the output of an env.observe as a 1x416 binary matrix.

    [card in front * 52]
    [card in mid * 52]
    [card in back * 52]
    [card in opponent front * 52]
    [card in opponent mid * 52]
    [card in opponent back * 52]
    [card in current card * 52]
    [card in remaining cards * 52]
    """
    def __init__(self):
        self.dim = 146

    def encode(plyr_board, oppo_board, current_card, plyr_cards):
        pass


class GamestateRankSuitEncoder(object):
    """Encode the output of an env.observe as a 1x63 integer matrix.
    Loses some info on the exact suit identity of cards, but much
    smaller than a GamestateBinaryEncoder representation.

    [rank of cards in front * 3]
    [rank of cards in mid * 5]
    [rank of cards in back * 5]
    [spade, heart, diamond, club count in front * 4]
    [spade, heart, diamond, club count in mid * 4]
    [spade, heart, diamond, club count in back * 4]
    [rank of cards in opponent front * 3]
    [rank of cards in opponent mid * 5]
    [rank of cards in opponent back * 5]
    [spade, heart, diamond, club count in opponent front * 4]
    [spade, heart, diamond, club count in opponent mid * 4]
    [spade, heart, diamond, club count in opponent back * 4]
    [rank of current card * 1]
    [spade, heart, diamond, club of current card * 4]
    [rank of remaining cards * 4]
    [spade, heart, diamond, club of remaining cards * 4]
    """
    def __init__(self):
        self.dim = 63

    def encode(plyr_board, oppo_board, current_card, plyr_cards):
        pass
