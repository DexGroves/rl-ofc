import collections
from treys import Card
from rlofc.rlofc.ofc_evaluator import OFCEvaluator

evaluator = OFCEvaluator()


class RoyaltyCalculator(object):
    """Score royalties for back, mid and front.
    I would love to do this with evaluator.get_rank_class, but it's
    broken: https://github.com/worldveil/deuces/issues/9
    """

    @staticmethod
    def score_back_royalties(cards):
        if len(cards) != 5:
            raise ValueError("Incorrect number of cards!")

        rank = evaluator.evaluate([], cards)

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
            
            return 25  # Royal flush 25

    @staticmethod
    def score_mid_royalties(cards):
        if len(cards) != 5:
            raise ValueError("Incorrect number of cards!")

        rank = evaluator.evaluate([], cards)

        if rank > 2467:
            return 0   # Nothing good enough

        if rank > 1609:
            return 2   # Three-of-a-kind

        return 2 * RoyaltyCalculator.score_back_royalties(cards)

    @staticmethod
    def score_front_royalties(cards):
        if len(cards) != 3:
            raise ValueError("Incorrect number of cards!")

        # Transfer cards into int
        ranks = [Card.get_rank_int(x) for x in cards]
        ctr = collections.Counter(ranks)
        # rank: card int, count: number of card
        # rank = 0, mean 2 in card
        rank, count = ctr.most_common()[0]

        if count < 2:
            return 0 # no pairs

        if count == 2:
            return max(0, rank - 3) # one pair

        if count == 3:
            return 10 + rank # Three-of-a-kind
