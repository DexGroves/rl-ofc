from rlofc.royalty_calculator import RoyaltyCalculator
from deuces import Card


no_pair  = [Card.new(x) for x in ['2h', '9c', '3d', '4d', '5d']]
one_pair = [Card.new(x) for x in ['Ah', 'Th', '8h', '5h', '5c']]
two_pair = [Card.new(x) for x in ['Ah', 'Th', 'Th', '5h', '5c']]
trips    = [Card.new(x) for x in ['Ah', 'Th', '8h', '8d', '8c']]
straight = [Card.new(x) for x in ['Ah', 'Kh', 'Qh', 'Jh', 'Tc']]
flush    = [Card.new(x) for x in ['Ah', 'Th', '8h', '5h', '4h']]
boat     = [Card.new(x) for x in ['Ah', 'As', 'Ac', '8d', '8c']]
quads    = [Card.new(x) for x in ['Ah', 'As', 'Ac', 'Ad', 'Kc']]
sf       = [Card.new(x) for x in ['Ah', '2h', '3h', '4h', '5h']]
royal    = [Card.new(x) for x in ['Ah', 'Kh', 'Qh', 'Jh', 'Th']]


def test_back_royalties():
    assert RoyaltyCalculator.score_back_royalties(no_pair) == 0
    assert RoyaltyCalculator.score_back_royalties(one_pair) == 0
    assert RoyaltyCalculator.score_back_royalties(two_pair) == 0
    assert RoyaltyCalculator.score_back_royalties(trips) == 0
    assert RoyaltyCalculator.score_back_royalties(straight) == 2
    assert RoyaltyCalculator.score_back_royalties(flush) == 4
    assert RoyaltyCalculator.score_back_royalties(boat) == 6
    assert RoyaltyCalculator.score_back_royalties(quads) == 10
    assert RoyaltyCalculator.score_back_royalties(sf) == 15
    assert RoyaltyCalculator.score_back_royalties(royal) == 35


def test_mid_royalties():
    assert RoyaltyCalculator.score_mid_royalties(no_pair) == 0
    assert RoyaltyCalculator.score_mid_royalties(one_pair) == 0
    assert RoyaltyCalculator.score_mid_royalties(two_pair) == 0
    assert RoyaltyCalculator.score_mid_royalties(trips) == 2
    assert RoyaltyCalculator.score_mid_royalties(straight) == 4
    assert RoyaltyCalculator.score_mid_royalties(flush) == 8
    assert RoyaltyCalculator.score_mid_royalties(boat) == 12
    assert RoyaltyCalculator.score_mid_royalties(quads) == 20
    assert RoyaltyCalculator.score_mid_royalties(sf) == 30
    assert RoyaltyCalculator.score_mid_royalties(royal) == 70


def test_front_royalties():
    front = [Card.new(x) for x in ['2h', '2c', '3d']]
    assert RoyaltyCalculator.score_front_royalties(front) == 0

    front = [Card.new(x) for x in ['6h', '6c', '3d']]
    assert RoyaltyCalculator.score_front_royalties(front) == 1

    front = [Card.new(x) for x in ['Ah', 'Ac', '3d']]
    assert RoyaltyCalculator.score_front_royalties(front) == 9

    front = [Card.new(x) for x in ['2h', '2c', '2d']]
    assert RoyaltyCalculator.score_front_royalties(front) == 10

    front = [Card.new(x) for x in ['Ah', 'Ac', 'Ad']]
    assert RoyaltyCalculator.score_front_royalties(front) == 22
