from rlofc.deck_generator import DeckGenerator
from rlofc.ofc_board import OFCBoard


class OFCEnvironment(object):
    """Handle OFC game state and rewards."""

    def __init__(self, lhs_agent, rhs_agent):
        self.lhs_agent = lhs_agent
        self.rhs_agent = rhs_agent

    def play_game(self):
        """Rollout one OFC game and return the LHS score and LHS/RHS boards."""
        deck = DeckGenerator.new_deck()

        lhs_board = OFCBoard()
        rhs_board = OFCBoard()

        lhs_start = deck[0:5]
        rhs_start = deck[6:11]

        # Starting hand one card at a time for now. In future, give
        # all cards at once
        for i in xrange(5):
            card = lhs_start[i]
            street_id = self.lhs_agent.place_new_card(card, lhs_board)
            lhs_board.place_card_by_id(card, street_id)

            card = rhs_start[i]
            street_id = self.rhs_agent.place_new_card(card, rhs_board)
            rhs_board.place_card_by_id(card, street_id)

        # Eight cards one at a time
        for i in xrange(8):
            card = deck.pop()
            street_id = self.lhs_agent.place_new_card(card, lhs_board)
            lhs_board.place_card_by_id(card, street_id)

            card = deck.pop()
            street_id = self.rhs_agent.place_new_card(card, rhs_board)
            rhs_board.place_card_by_id(card, street_id)

        lhs_royalties = lhs_board.get_royalties()
        rhs_royalties = rhs_board.get_royalties()

        if lhs_board.is_foul() and rhs_board.is_foul():
            lhs_score = 0

        elif lhs_board.is_foul():
            lhs_score = (-1 * rhs_royalties) - 6

        elif rhs_board.is_foul():
            lhs_score = lhs_royalties + 6

        else:
            exch = self.calculate_scoop(lhs_board,
                                        rhs_board)
            lhs_score = exch + lhs_royalties - rhs_royalties

        return lhs_score, lhs_board, rhs_board

    def calculate_scoop(self, lhs_board, rhs_board):
        lhs_won = 0

        lhs_won += self.calculate_street(lhs_board.front, rhs_board.front)
        lhs_won += self.calculate_street(lhs_board.mid, rhs_board.mid)
        lhs_won += self.calculate_street(lhs_board.back, rhs_board.back)

        if lhs_won in [3, -3]:   # Scoop, one way or the other
            lhs_won = lhs_won * 2

        return lhs_won

    @staticmethod
    def calculate_street(lhs_hand, rhs_hand):
        lhs_rank = lhs_hand.get_rank()
        rhs_rank = rhs_hand.get_rank()

        if lhs_rank < rhs_rank:
            return 1
        if rhs_rank < lhs_rank:
            return -1
        return 0
