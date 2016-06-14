from rlofc.ofc_game import OFCGame
from rlofc.ofc_policy import OFCRandomPolicy


def test_ofc_game_runs():
    lhs = OFCRandomPolicy()
    rhs = OFCRandomPolicy()

    ofc_game = OFCGame(lhs, rhs)

    # ofc_game.lhs_policy.board.pretty()
    # ofc_game.rhs_policy.board.pretty()
    assert type(ofc_game.play_game()) is int
