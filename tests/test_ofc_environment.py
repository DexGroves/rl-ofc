from rlofc.ofc_environment import OFCEnvironment
from rlofc.ofc_agent import OFCRandomAgent


def test_ofc_game_runs():
    lhs = OFCRandomAgent()
    rhs = OFCRandomAgent()

    ofc_game = OFCEnvironment(lhs, rhs)

    score, lhs_board, rhs_board = ofc_game.play_game()
    # score
    # lhs_board.pretty()
    # rhs_board.pretty()
    assert type(score) is int
