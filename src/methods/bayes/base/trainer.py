
class ModelDistribution: ...


class BaseBayesTrainer:
    def train(self, *args, **kwargs) -> ModelDistribution:
        ...   
