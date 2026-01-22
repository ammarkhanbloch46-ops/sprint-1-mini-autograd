∂L/∂w = ∂L/∂y × ∂y/∂w
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = []

    def __add__(self, other):
        out = Value(self.data + other.data)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def backward(self):
        self.grad = 1.0
        self._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"



