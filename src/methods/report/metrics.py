class BaseMetric:
    def __call__(self, accuracy: float):
        print(accuracy)
