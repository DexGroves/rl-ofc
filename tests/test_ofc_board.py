from rlofc.ofc_board import OFCBoard


def test_is_complete():
    board = OFCBoard()

    board.set_front(['6s', '6d', '5s'])
    board.set_mid(['9d', '9c', '9s', '2d', '3d'])
    board.set_back(['Ah', '2h', '3h', '4h'])

    assert not board.is_complete()

    board.set_back(['Ah', '2h', '3h', '4h', '5h'])

    assert board.is_complete()


def test_get_royalties():
    board = OFCBoard()

    board.set_front(['6s', '6d', '5s'])
    board.set_mid(['9d', '9c', '9s', '2d', '3d'])
    board.set_back(['Ah', '2h', '3h', '4h', '5h'])

    assert board.get_royalties() == 18


def test_is_foul():
    board = OFCBoard()

    board.set_front(['6s', '6d', '5s'])
    board.set_mid(['6d', '6c', '4s', '2d', '3d'])
    board.set_back(['Ah', '2h', '3h', '4h', '5h'])

    assert board.is_foul()

    board.set_front(['6s', '6d', '5s'])
    board.set_mid(['6d', '6c', '9s', '2d', '3d'])
    board.set_back(['Ah', '2h', '3h', '4h', '5h'])

    assert not board.is_foul()


def test_available_streets():
    board = OFCBoard()

    board.set_front(['6s', '6d', '5s'])
    board.set_mid(['6d', '6c', '4s', '2d', '3d'])
    board.set_back(['Ah', '2h', '3h', '4h', '5h'])

    assert board.get_free_streets() == [0, 0, 0]

    board.set_front(['6s', '6d'])
    board.set_mid(['6d', '6c', '2d', '3d'])
    board.set_back(['Ah', '2h', '3h', '4h', '5h'])

    assert board.get_free_streets() == [1, 1, 0]
