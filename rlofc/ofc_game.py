from rlofc.deck_generator import DeckGenerator


class OFCGame(object):
    """Play out OFC games between two policies."""

    def __init__(self, lhs_policy, rhs_policy):
        self.lhs_policy = lhs_policy
        self.rhs_policy = rhs_policy

    def play_game(self):
        """Rollout one OFC game and return the LHS score."""
        deck = DeckGenerator.new_deck()

        self.lhs_policy.board.clear()
        self.rhs_policy.board.clear()

        lhs_start = deck[0:5]
        rhs_start = deck[6:11]

        self.lhs_policy.place_starting_hand(lhs_start)
        self.rhs_policy.place_starting_hand(rhs_start)

        for i in xrange(8):
            self.lhs_policy.place_new_card(deck.pop())
            self.rhs_policy.place_new_card(deck.pop())

        lhs_royalties = self.lhs_policy.board.get_royalties()
        rhs_royalties = self.lhs_policy.board.get_royalties()

        if self.lhs_policy.board.is_foul() and self.rhs_policy.board.is_foul():
            return 0

        if self.lhs_policy.board.is_foul():
            return rhs_royalties - 6

        if self.rhs_policy.board.is_foul():
            return lhs_royalties + 6

        else:
            exch = self.calculate_scoop(self.lhs_policy.board,
                                        self.rhs_policy.board)
            return exch + lhs_royalties - rhs_royalties

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
