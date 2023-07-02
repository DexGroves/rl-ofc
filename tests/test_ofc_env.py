import random
from rlofc.rlofc.ofc_environment import OFCEnv


def test_ofc_env():
    env = OFCEnv([])
    my_board, oppo_board, cur_card, cards, game_over, score = env.observe()
    while not game_over:
        decision = random.choice(my_board.get_free_street_indices())
        env.step(decision)
        my_board, oppo_board, cur_card, cards, game_over, score = env.observe()

    # my_board.pretty()
    # oppo_board.pretty()
    assert type(score) is int
