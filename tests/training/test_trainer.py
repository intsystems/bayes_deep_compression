""" Unit testing of trainer classes 
"""
import pytest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from src.methods.bayes.variational.net import LogUniformVarLayer
from src.methods.bayes.variational.optimization import LogUniformVarKLLoss
from src.methods.bayes.variational.trainer import *
from src.methods.report.base import ReportChain  # Это просто список callback
# Этот модуль callback просто выводит каждый шаг данные от тренера
from src.methods.report.variational import VarBaseReport


def test_simple_trainer(mnist_classifier: nn.Module, mnist_dataset: Dataset):
    beta = Beta_Scheduler_Plato()

    EPOCHS = 2
    BATCH_SIZE = 4
    LR = 1e-0  # 5e-4
    # Split the training set into training and validation sets
    VAL_PERCENT = 0.2  # percentage of the data used for validation
    SAMPLES = 2
    BETA = 0.01  # 5e-5 #0.01
    BETA_FAC = 5e-1
    PRUNE = 1.9  # 1.99, 2.1, 1.9
    PLATO_TOL = 20

    # copy inital paramter's values of the classifier
    init_model_params = dict(mnist_classifier.named_parameters())
    init_model_params = {param_name: param.detach()
                         for param_name, param in init_model_params.items()}

    # define bayes models
    var_module = LogUniformVarLayer(mnist_classifier)
    model = VarBayesNet(mnist_classifier, nn.ModuleList([var_module]))

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Первый лосс это обычный лосс на данные, второй лосс это лосс байесковской модели
    fit_loss = nn.CrossEntropyLoss()
    kl_loss = LogUniformVarKLLoss()

    # Используем планировщик коэффицента пропорциональности между fit_loss и kl_loss
    beta = Beta_Scheduler_Plato(BETA, BETA_FAC, PLATO_TOL)
    beta_KL = Beta_Scheduler_Plato(
        beta.beta, 1 / BETA_FAC, PLATO_TOL, ref=beta, threshold=1e-4)

    # Данная функция будет выполнятся после каждого шага тренера, соответсвенно нам требуется сделать шаг планировщика и изменить соотвествующий коэффициент
    def post_train_step(trainer: VarTrainerParams, train_result: VarBayesTrainer.TrainResult):
        beta.step(train_result.fit_loss)
        beta_KL.step(train_result.dist_loss)
        trainer.params.beta = float(beta)

    val_size = int(VAL_PERCENT * len(mnist_dataset))
    train_size = len(mnist_dataset) - val_size

    t_dataset, v_dataset = torch.utils.data.random_split(mnist_dataset,
                                                         [train_size,
                                                          val_size])

    # Create DataLoaders for the training and validation sets
    train_loader = torch.utils.data.DataLoader(t_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(v_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              pin_memory=True)

    accuracy_collector = CallbackLossAccuracy()

    train_params = VarTrainerParams(EPOCHS, optimizer, fit_loss, kl_loss, SAMPLES, PRUNE, BETA, {
                                    'accuracy': accuracy_collector})

    # Если хотим сделать бету фиксированной, то нунжо убрать аргумент [post_train_step]
    trainer = VarBayesTrainer(train_params, ReportChain(
        [VarBaseReport()]), train_loader, eval_loader, [post_train_step])
    trained_model = trainer.train(model)

    # check callback correctness
    assert 0 <= accuracy_collector.sum_acc / accuracy_collector.samples <= 1

    # check that model weights has been changed
    trained_model_params = dict(trained_model.get_model().named_parameters())
    for param_name in init_model_params:
        assert not torch.allclose(
            init_model_params[param_name],
            trained_model_params[param_name]
        )

    # check eval_thresholds correctness
    for eval_res in trainer.eval_thresholds(trained_model, np.linspace(-1, 1, 5).tolist()):
        assert 0 <= eval_res.cnt_prune_parameters <= eval_res.cnt_params
        assert np.allclose(eval_res.val_loss,
                           eval_res.fit_loss + eval_res.dist_loss)
