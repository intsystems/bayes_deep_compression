from src.components.datasets.mnist import MNISTDatasetLoader
from src.methods.bayes.kf_laplace.trainer import KFEagerTraining, KfTrainerParams
from src.components.nets.kf_net import KfMLPComponent
from torch.optim.sgd import SGD
from src.components.report.dummy import DummyReportChain

kf_distribution = KFEagerTraining(
    model=KfMLPComponent(input_dim=64, output_dim=10),
    dataset_loader=MNISTDatasetLoader(),
    params=KfTrainerParams(
        num_epochs=1,
        optimizer=SGD,
    ),
    report_chain=DummyReportChain(),
).train()

kf_distribution.sample_net()
