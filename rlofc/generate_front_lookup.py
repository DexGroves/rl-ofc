"""Grizzly, grizzly, grizzly hacks to make a three-card hand fit with
the mechanics of deuces. Extends ranks into floats to allow fronts to
be compared with each other and with mids.
"""


from copy import copy
from deuces import Card, Evaluator
import cPickle as pickle


def get_lowest_unpairing_card(hand):
    """Add the worst possible card to an incomplete hand. Something
    that cannot pair up or make a flush or make a straight."""
    existing_ranks = set([Card.get_rank_int(x) for x in hand])
    remaining_ranks = list(set(range(12)) - existing_ranks)
    selected_rank = remaining_ranks[0]

    would_be_hand = hand + [Card.new(Card.STR_RANKS[selected_rank] + 'h')]
    if is_straight(would_be_hand):
        selected_rank = remaining_ranks[1]    # Don't make a straight

    selected_rank_str = Card.STR_RANKS[selected_rank]

    last_suit = Card.get_suit_int(hand[-1])
    last_suit_index = [1, 2, 4, 8].index(last_suit)
    selected_suit = [1, 2, 4, 8][(last_suit_index + 1) % 4]
    selected_suit_str = Card.INT_SUIT_TO_CHAR_SUIT[selected_suit]

    selected_card_str = selected_rank_str + str(selected_suit_str)
    return Card.new(selected_card_str)


def is_straight(cards):
    if len(cards) < 5:
        return False

    rank = evaluator.evaluate(cards, [])
    rank_class = evaluator.get_rank_class(rank)

    if rank_class in [0, 1, 5]:
        return True

    return False


evaluator = Evaluator()
cards = ['Ah', 'Kh', 'Qh', 'Jh', 'Th', '9h', '8h',
         '7h', '6h', '5h', '4h', '3h', '2h']

perms = []
for i in xrange(len(cards)):
    for j in xrange(i, len(cards)):
        for k in xrange(j, len(cards)):
            perms.append([Card.new(cards[i]),
                          Card.new(cards[j]),
                          Card.new(cards[k])])

front_lookup = {}
for perm in perms:
    # Add the two lowest unpairing cards
    hand = copy(perm)

    hand.append(get_lowest_unpairing_card(hand))
    hand.append(get_lowest_unpairing_card(hand))

    prime_prod = Card.prime_product_from_hand(perm)
    rank = evaluator.evaluate(hand, []) + 1
    kicker = Card.get_rank_int(perm[0]) * 0.01 + \
        Card.get_rank_int(perm[1]) * 0.0001 + \
        Card.get_rank_int(perm[2]) * 0.000001

    front_lookup[prime_prod] = rank - kicker

with open('res/front_lookup.p', 'wb') as f:
    pickle.dump(front_lookup, f)
