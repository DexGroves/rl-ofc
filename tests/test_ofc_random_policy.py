from rlofc.ofc_policy import OFCRandomPolicy
from rlofc.deck_generator import DeckGenerator


def test_ofc_random_policy_can_complete():
    policy = OFCRandomPolicy()

    deck = DeckGenerator.new_deck()
    starting_hand = deck[46:51]

    policy.place_starting_hand(starting_hand)
    policy.place_new_card(deck.pop())
    policy.place_new_card(deck.pop())
    policy.place_new_card(deck.pop())
    policy.place_new_card(deck.pop())
    policy.place_new_card(deck.pop())
    policy.place_new_card(deck.pop())
    policy.place_new_card(deck.pop())
    policy.place_new_card(deck.pop())

    # policy.board.pretty()
    assert policy.board.is_complete()
