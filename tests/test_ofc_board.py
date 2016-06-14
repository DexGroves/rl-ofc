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
