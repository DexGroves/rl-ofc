import cPickle as pickle
from deuces import Card
from rlofc.ofc_evaluator import OFCEvaluator


front_lookup = pickle.load(open("res/front_lookup.p"))
evaluator = OFCEvaluator()


def test_front_lookup_aces():
    strongest = [Card.new(x) for x in ['Ah', 'As', '6d']]
    middle = [Card.new(x) for x in ['Ah', 'As', '5d', '4d', '3d']]
    weakest = [Card.new(x) for x in ['Ah', 'As', '5d']]

    assert evaluator.evaluate(strongest, []) < \
        evaluator.evaluate(middle, []) < \
        evaluator.evaluate(weakest, [])


def test_front_lookup_trips():
    weakest = [Card.new(x) for x in ['Kh', 'Ks', 'Kd', 'Ad', 'Qh']]
    middle = [Card.new(x) for x in ['Ah', 'As', 'Ad']]
    strongest = [Card.new(x) for x in ['Ah', 'As', 'Ad', '2d', '3d']]

    assert evaluator.evaluate(strongest, []) < \
        evaluator.evaluate(middle, []) < \
        evaluator.evaluate(weakest, [])


def test_front_lookup_very_low():
    weakest = [Card.new(x) for x in ['2h', '3s', '4d']]
    middle = [Card.new(x) for x in ['2h', '3s', '5d']]
    stronger = [Card.new(x) for x in ['2h', '4s', '5d']]
    strongest = [Card.new(x) for x in ['2h', '3s', '7d']]

    assert evaluator.evaluate(strongest, []) < \
        evaluator.evaluate(stronger, []) < \
        evaluator.evaluate(middle, []) < \
        evaluator.evaluate(weakest, [])
