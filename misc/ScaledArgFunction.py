from cil.optimisation.functions import Function
class ScaledArgFunction(Function):
    def __init__(self, function, scalar):
        self.function = function
        self.scalar = scalar

    def __call__(self, x):
        x *= self.scalar
        ret = self.function(x)
        x /= self.scalar
        return ret

    def gradient(self, x, out=None):
        x *= self.scalar
        should_return = False
        if out is None:
            out = self.function.gradient(x)
            should_return = True
        else:
            self.function.gradient(x, out=out)
        out *= self.scalar
        x /= self.scalar
        if should_return:
            return out

    def proximal(self, x, tau, out=None):
        # eq 6.6 of https://archive.siam.org/books/mo25/mo25_ch6.pdf
        should_return = False
        if out is None:
            out = x * 0
            should_return = True
        x *= self.scalar
        self.function.proximal( x, tau * self.scalar**2, out=out)
        x /= self.scalar
        out /= self.scalar
        if should_return:
            return out
    def convex_conjugate(self, x):
        # https://en.wikipedia.org/wiki/Convex_conjugate#Table_of_selected_convex_conjugates
        x /= self.scalar
        ret = self.function.convex_conjugate(x)
        x *= self.scalar
        return ret