from src.methods.bayes.kf_laplace.trainer import KFEagerTraining


class KFEagerTrainingComponent(KFEagerTraining):
    def __init__(self, model, visualizer):
        super().__init__(model, visualizer)
