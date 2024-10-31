""" draft ideas for tests
"""

bayes_nn = MakeModuleBayessian(
    ClassificationNet(IMG_SIZE, NUM_CLASSES, exp_params["num_layers"], exp_params["hidden_size"])
)

for module in bayes_nn.children():
    print(module)

imgs, _ = next(iter(train_loader))
imgs = imgs.to(dtype=torch.float32)
a = bayes_nn(imgs)
b = bayes_nn(imgs)
print(torch.allclose(a, b, rtol=1e-3))
print(a.flatten()[0], b.flatten()[0])

samples = bayes_nn.sample_estimation(imgs, 5)
print(len(samples))
print(samples[0].shape)

map_estimation = bayes_nn.get_map_module(True)