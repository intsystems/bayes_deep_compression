from functools import reduce

class Tracker:
    def __init__(self, generator):
        self.generator = generator
        self.cnt =0 

    def __iter__(self):
        return self

    def __next__(self):
        self.cnt += 1
        return next(self.generator)


tracked_generator = Tracker((t for t in range(10)) for _ in range(10))
print(
    reduce(lambda u,v: [u_elem + v_elem for u_elem, v_elem in zip(u,v)],tracked_generator, list(next(tracked_generator)))
)
print(tracked_generator.cnt)