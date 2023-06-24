import numpy as np
from deuces import Card


class GamestateEncoder(object):
    """Generic gamestate encoder methods."""

    def __init__(self):
        self.dim = None

    @staticmethod
    def cards_to_ranks(cards, pad):
        rank_dummy = np.zeros(pad)
        # Add one to distinguish deuce from missing
        cards = sorted([Card.get_rank_int(x) for x in cards])
        for i, card in enumerate(cards):
            if i >= pad:
                continue
            rank_dummy[i] = card + 1
        #? What is Hacky standardisation, according to the follow scripts, we can see it imply the mean is 8, and standard deviation is 14.
        rank_dummy_std = (rank_dummy - 8) / 14  # Hacky "standardisation"
        return rank_dummy_std

    @staticmethod
    def cards_to_suits(cards):
        suit_dummy = np.zeros(4)
        suit_binaries = np.array([Card.get_suit_int(x) for x in cards])
        # 1: spades
        # 2: hearts
        # 4: diamonds
        # 8: clubs
        suit_dummy[0] = sum(suit_binaries == 1)
        suit_dummy[1] = sum(suit_binaries == 2)
        suit_dummy[2] = sum(suit_binaries == 4)
        suit_dummy[3] = sum(suit_binaries == 8)
        #? According to the scripts below, we can see that it imply the mean is 1.5, and standard deviation is 2.
        suit_dummy_std = (suit_dummy - 1.5) / 2  # Hacky "standardisation"
        return suit_dummy_std

    @staticmethod
    def card_to_ranks_binary(card):
        out = np.zeros(13)
        rank = Card.get_rank_int(card)
        out[rank] = 1
        return out


class GamestateBinaryEncoder(GamestateEncoder):
    """Encode the output of an env.observe as a 1x416 binary matrix.

    [card in front * 52]:the front line of myself
    [card in mid * 52]: the middle line of myself
    [card in back * 52]: the back line of my self
    [card in opponent front * 52]: the fron line of the oppo
    [card in opponent mid * 52]: the mid line of the oppo
    [card in opponent back * 52]: the back line of the oppo
    [card in current card * 52]: all the cards appear on the table
    [card in remaining cards * 52]: the remaining cards not on the table
    """
    def __init__(self):
        self.dim = 146

    def encode(plyr_board, oppo_board, current_card, plyr_cards):
        pass


class GamestateRankSuitEncoder(GamestateEncoder):
    """Encode the output of an env.observe as a 1x63 integer matrix.
    Loses some info on the exact suit identity of cards, but much
    smaller than a GamestateBinaryEncoder representation.

    [rank of cards in front * 3]: 前排放 3 張
    [rank of cards in mid * 5]: 中排放 5 張
    [rank of cards in back * 5]: 後排放 5 張
    [spade, heart, diamond, club count in front * 4]: 計算前排花色
    [spade, heart, diamond, club count in mid * 4]: 計算中排花色
    [spade, heart, diamond, club count in back * 4]: 計算後排花色
    [rank of cards in opponent front * 3]: 對手前排 3 張
    [rank of cards in opponent mid * 5]: 對手中排 5 張
    [rank of cards in opponent back * 5]: 對手後排 5 張
    [spade, heart, diamond, club count in opponent front * 4]: 計算對手前排花色
    [spade, heart, diamond, club count in opponent mid * 4]: 計算對手中排花色
    [spade, heart, diamond, club count in opponent back * 4]: 計算對手後排花色
    [rank of current card * 1]: 開牌第一張數字
    [spade, heart, diamond, club of current card * 4]: 開牌第一章的花色
    [rank of remaining cards * 4]: 剩餘 4 張數字
    [spade, heart, diamond, club of remaining cards * 4]: 剩餘 4 張的花色
    [free street binaries]
    """
    def __init__(self):
        self.dim = 66

    def encode(self, plyr_board, oppo_board, current_card,
               plyr_cards, game_over, score):
        current_card = [Card.new(current_card)]
        plyr_cards = [Card.new(x) for x in plyr_cards]

        plyr_front_ranks = self.cards_to_ranks(plyr_board.front.cards, 3)
        plyr_mid_ranks = self.cards_to_ranks(plyr_board.mid.cards, 5)
        plyr_back_ranks = self.cards_to_ranks(plyr_board.back.cards, 5)

        oppo_front_ranks = self.cards_to_ranks(plyr_board.front.cards, 3)
        oppo_mid_ranks = self.cards_to_ranks(plyr_board.mid.cards, 5)
        oppo_back_ranks = self.cards_to_ranks(plyr_board.back.cards, 5)

        plyr_front_suits = self.cards_to_suits(plyr_board.front.cards)
        plyr_mid_suits = self.cards_to_suits(plyr_board.mid.cards)
        plyr_back_suits = self.cards_to_suits(plyr_board.back.cards)

        oppo_front_suits = self.cards_to_suits(plyr_board.front.cards)
        oppo_mid_suits = self.cards_to_suits(plyr_board.mid.cards)
        oppo_back_suits = self.cards_to_suits(plyr_board.back.cards)

        current_card_rank = self.cards_to_ranks(current_card, 1)
        current_card_suit = self.cards_to_suits(current_card)

        remaining_card_ranks = self.cards_to_ranks(plyr_cards, 4)
        remaining_card_suits = self.cards_to_suits(plyr_cards)

        free_streets = np.array(plyr_board.get_free_streets())
        free_streets_std = (free_streets - 0.5) * 2  # Hacky "standardisation"

        encoding = np.hstack([
            plyr_front_ranks,
            plyr_mid_ranks,
            plyr_back_ranks,
            plyr_front_suits,
            plyr_mid_suits,
            plyr_back_suits,
            oppo_front_ranks,
            oppo_mid_ranks,
            oppo_back_ranks,
            oppo_front_suits,
            oppo_mid_suits,
            oppo_back_suits,
            current_card_rank,
            current_card_suit,
            remaining_card_ranks,
            remaining_card_suits,
            free_streets_std
        ])

        return encoding


class GamestateStreetsonlyEncoder(GamestateEncoder):
    """Just return which streets are open."""

    def __init__(self):
        self.dim = 3

    def encode(self, plyr_board, oppo_board, current_card,
               plyr_cards, game_over, score):
        free_streets = np.array(plyr_board.get_free_streets())
        free_streets_std = (free_streets - 0.5) * 2  # Hacky "standardisation"

        return np.array(free_streets_std)


class GamestateSelfranksonlyEncoder(GamestateEncoder):
    """Return only self ranks."""

    def __init__(self):
        self.dim = 17

    def encode(self, plyr_board, oppo_board, current_card,
               plyr_cards, game_over, score):
        current_card = [Card.new(current_card)]
        plyr_front_ranks = self.cards_to_ranks(plyr_board.front.cards, 3)
        plyr_mid_ranks = self.cards_to_ranks(plyr_board.mid.cards, 5)
        plyr_back_ranks = self.cards_to_ranks(plyr_board.back.cards, 5)

        current_card_rank = self.cards_to_ranks(current_card, 1)

        free_streets = np.array(plyr_board.get_free_streets())
        free_streets_std = (free_streets - 0.5) * 2  # Hacky "standardisation"

        encoding = np.hstack([
            plyr_front_ranks,
            plyr_mid_ranks,
            plyr_back_ranks,
            current_card_rank,
            free_streets_std
        ])

        return encoding


class SelfRankBinaryEncoder(GamestateEncoder):
    """Return self rank information in binary form."""

    def __init__(self):
        self.dim = 16

    def encode(self, plyr_board, oppo_board, current_card,
               plyr_cards, game_over, score):
        if current_card is not None:
            current_card = Card.new(current_card)
            current_card_binary = self.card_to_ranks_binary(current_card)
        else:
            current_card_binary = np.zeros(13)

        free_streets = np.array(plyr_board.get_free_streets())
        free_streets_std = (free_streets - 0.5) * 2  # Hacky "standardisation"

        encoding = np.hstack([
            current_card_binary,
            free_streets_std
        ])

        return encoding
