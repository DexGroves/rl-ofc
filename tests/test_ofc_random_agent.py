from rlofc.ofc_board import OFCBoard
from rlofc.ofc_agent import OFCRandomAgent
from rlofc.deck_generator import DeckGenerator


def test_ofc_random_policy_can_complete():
    agent = OFCRandomAgent()

    deck = DeckGenerator.new_deck()
    board = OFCBoard()

    for i in xrange(13):
        draw = deck.pop()
        street_id = agent.place_new_card(draw, board)
        board.place_card_by_id(draw, street_id)

    # board.pretty()
    assert board.is_complete()
