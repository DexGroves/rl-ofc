from rlofc.royalty_calculator import RoyaltyCalculator
from deuces import Card


def test_five_card_royalties():
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
