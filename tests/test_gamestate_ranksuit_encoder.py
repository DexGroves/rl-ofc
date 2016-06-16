from rlofc.ofc_environment import OFCEnv
from rlofc.gamestate_encoder import GamestateRankSuitEncoder


def test_gamestate_ranksuit_encoder():
    env = OFCEnv([])
    encoder = GamestateRankSuitEncoder()

    plyr_board, oppo_board, cur_card, cards, game_over, score = env.observe()

    encoding = encoder.encode(*env.observe())

    for num in encoding:
        assert num is not None
