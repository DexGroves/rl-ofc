import _pickle as pickle
from treys import Card, Evaluator
from treys.lookup import LookupTable

class StrToBytes:
    def __init__ (self, fileobj):
        self.fileobj = fileobj
        
    def read(self, size):
        return self.fileobj.read(size).encode()
    
    def readline (self, size=-1):
        return self.fileobj. readline(size).encode()
    
FRONT_LOOKUP = pickle.load(StrToBytes(open("rlofc/res/front_lookup.p", 'r')))


# FRONT_LOOKUP = pickle.load(open("res/front_lookup.p"))


class OFCEvaluator(Evaluator):
    """ treys' evaluator class extended to score an OFC Front."""
    def __init__(self):
        self.table = LookupTable()

        self.hand_size_map = {
            3: self._three,
            5: self._five,
            6: self._six,
            7: self._seven,
            8: self._fantasy,
            9: self._fantasy,
            10: self._fantasy,
            11: self._fantasy,
            12: self._fantasy,
            13: self._fantasy,
            14: self._fantasy,
            15: self._fantasy,
            16: self._fantasy
        }

    def _three(self, cards):
        prime = Card.prime_product_from_hand(cards)
        return FRONT_LOOKUP[prime]
    
    def _fantasy(self, cards):
        
        return 0
