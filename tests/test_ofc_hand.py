from rlofc.ofc_board import OFCHand


def test_five_card_royalties():
    no_pair  = OFCHand(['2h', '2c', '3d', '4d', '5d'], 5)
    one_pair = OFCHand(['Ah', 'Th', '8h', '5h', '5c'], 5)
    two_pair = OFCHand(['Ah', 'Th', 'Th', '5h', '5c'], 5)
    trips    = OFCHand(['Ah', 'Th', '8h', '8d', '8c'], 5)
    straight = OFCHand(['Ah', 'Kh', 'Qh', 'Jh', 'Tc'], 5)
    flush    = OFCHand(['Ah', 'Th', '8h', '5h', '4h'], 5)
    boat     = OFCHand(['Ah', 'As', 'Ac', '8d', '8c'], 5)
    quads    = OFCHand(['Ah', 'As', 'Ac', 'Ad', 'Kc'], 5)
    sf       = OFCHand(['Ah', '2h', '3h', '4h', '5h'], 5)
    royal    = OFCHand(['Ah', 'Kh', 'Qh', 'Jh', 'Th'], 5)

    assert no_pair.get_royalties() == 0
    assert one_pair.get_royalties() == 0
    assert two_pair.get_royalties() == 0
    assert trips.get_royalties() == 0
    assert straight.get_royalties() == 2
    assert flush.get_royalties() == 4
    assert boat.get_royalties() == 6
    assert quads.get_royalties() == 10
    assert sf.get_royalties() == 15
    assert royal.get_royalties() == 35
