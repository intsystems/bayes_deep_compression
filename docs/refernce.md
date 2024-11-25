## Обучение байесовской модели на основе модели PyTorch
Библиотека работает на основе pythorch. Поэтому для начала нужно создать модель из pytorch, которую хотим обучить 
```python
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import sys
```

Создаем простой классификтор, который будет нашей базовой моделью, кторую мы хотим обучить и запрунить


```python

class Classifier(nn.Module): 
    def __init__(self, classes: int = 10): 
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        #self.dropout1 = nn.Dropout2d(0.25) 
        #self.dropout2 = nn.Dropout2d(0.5) 
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, classes) 
  
    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        #x = self.dropout1(x) 
        x = self.pool(F.relu(self.conv2(x))) 
        #x = self.dropout2(x) 
        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x
```

Загружаем датасет MNIST, на котором мы хотим обучить наш классификатор


```python
test_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
```

Далее необходимо превраитить модель в байесовскую. Для этого необходимо обернуть в BayesLayer (LogUniformVarLayer) слои, для которых мы хотим применить байесовское обучение. А также BayesNet (VarBayesNet), которая хранит все байесовские слои и изначальную сеть.
Для выбора конкретного метода обучения нужно выбрать BaseLoss(LogUniformVarKLLoss), который умеет работать с выбранными слоями.

## Создание байесовской модели на основе nn.Module
```python
from src.methods.bayes.variational.net import LogUniformVarLayer, VarBayesNet #Первым модулоем мы оборачиваем те слои модели, которые мы хотим сделать байесовыми, второй модуль это сама байесовская сеть
from src.methods.bayes.variational.optimization import LogUniformVarKLLoss #Это лосс байесовской модели, который отвечает за тип обучения. Всегда рекомендуется использовать специализированный лосс, но для большинства распределений его нет
```


Первым делом создадим нашу базовую модель


```python
module = Classifier()
```

Далее мы часть слоев превратим в байесовски с помощью LogUniformVarBayesModule. И создадим список всех слоев nn.ModuleList([layer1, layer2, ...]), которые мы хотим обучить (в том чилсе слои, которые не являются байесовыми). Заметим, что можно обернуть и всю сеть целиком и передать список состоящий только из нее.


```python
var_module1 = LogUniformVarLayer(module.conv1)
bayes_model = VarBayesNet(module, nn.ModuleDict({'conv1': var_module1}))
```

## Пример шага обучения
Посомотрим как выглядит шаг обучения для сети.

```python
optimizer = optim.Adam(bayes_model.parameters(), lr=1e-3)
```


В целом он ничем не отличается от обычного шага, нам только нужно парвильно агрегировать лоссы от нескольких семплов на одном шаге


```python
#get one sample
#========
image, label = test_dataset[10]
y = bayes_model(torch.ones_like(image))
kl_loss = LogUniformVarKLLoss()
#========

#list of fit_loss for each sample (we have one sample)
fit_loss = [y.sum()]
 #list of dist_loss for each sample (we have one sample)
dist_loss = [kl_loss(posterior = bayes_model.posterior, prior = bayes_model.prior, param_sample_dict = bayes_model.weights)]
beta = 0.1 # scale factor betwenn dist_loss and data_loss
#aggregation result is stored in total_loss attribute, all others are provided for statistic of traininghow important each part is
aggregation_result = kl_loss.aggregate(fit_loss, dist_loss, beta) 
out = aggregation_result.total_loss # calculated loss for one step
#optimizer step
optimizer.zero_grad() 
out.backward() 
optimizer.step() 
```

Создать распределение сетей можно просто из распределения на параметры и базовой сети


```python
net_distributon = VarBayesModuleNetDistribution(bayes_model.base_module, bayes_model.posterior)
#Это прунер, которые зануляет веса в зависимости от плотности распределения при 0
net_distributon_pruner = BaseNetDistributionPruner(net_distributon)
#Здесь мы устанавливаем средние веса модели  
net_distributon.set_map_params()
#Пруним на основе определенного порога
net_distributon_pruner.prune(1.9)
#get basic model for evaluation
eval_model = net_distributon.get_model()
```

Мы получили модель с той же архитектурой что и изначальная


```python
print(eval_model.conv1.weight)
```

```python
print(bayes_model.state_dict())
```


Forward делается по последнему сохраненному сэмплу. Заметим, что мы нигде не копируем данные, и модели не инкапсулируется. Поэтому, чтобы отвязать, их неободимо скопировать


```python
print(bayes_model(torch.zeros_like(image)))
#print(bayes_model(torch.zeros_like(image), sample = False))
print(module(torch.zeros_like(image)))
```

Для обучения рекомендуется использовать GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda')



## Обучение с помощью встроенного тренера
Далее мы импортируем несколько модулей для обучения

Сам тренер, Параметры тренера, Планировщик beta(коэффициент сооьношения между обычным лоссом и байесовским), и callback для метрики точности
```python
from src.methods.bayes.variational.trainer import VarBayesTrainer, VarTrainerParams, Beta_Scheduler_Plato, CallbackLossAccuracy
```
Список callbacks
```python
from src.methods.report.base import ReportChain
```
И для примера какой-нибудь callback. Этот модуль callback просто выводит каждый шаг данные от тренера

```python
from src.methods.report.variational import VarBaseReport 
```



Инициализируем трйенер. Вам не обязательно писать свой тренер. Для всех вариационных методов уже есть готовый

Задаем сначала парметрвы обучения
```python
BATCH_SIZE=1000
EPOCHS=4000
LR = 1e-3 #5e-4
# Split the training set into training and validation sets 
VAL_PERCENT = 0.2 # percentage of the data used for validation 
SAMPLES = 10
BETA = 0.01 #5e-5 #0.01
BETA_FAC = 5e-1
PRUNE = 1.9#1.99, 2.1, 1.9
PLATO_TOL = 20

train_params = VarTrainerParams(EPOCHS, optimizer,fit_loss, kl_loss, SAMPLES, PRUNE, BETA, {'accuracy': CallbackLossAccuracy()})
```

Потом создаем байесвскую сеть на основе обычной
```python
base_module = Classifier()
var_module1 = LogUniformVarLayer(base_module.conv1)
#Первый аргумент базовая сеть, второй список всех слоев (где нужные из них являются байесовыми)
model = VarBayesNet(base_module, nn.ModuleDict({'conv1': var_module1}))
```

Выбираем оптимизатор, который хотим использовать для задачи
```python
optimizer = optim.Adam(model.parameters(), lr=LR)
```
Выбираем loss который мв хотим использовать. Он должен быть совместим с модулем который мы используем. Но заметим, что не все модули совмести со всеми loss, некторые loss являются специфическими для определенных модулей
```python
#Первый лосс это обычный лосс на данные, второй лосс это лосс байесковской модели
fit_loss = nn.CrossEntropyLoss() 
kl_loss = LogUniformVarKLLoss()
```
Для ставбильности обучения рекомендуется использовать планирофшик beta
```python
#Используем планировщик коэффицента пропорциональности между fit_loss и kl_loss
beta = Beta_Scheduler_Plato(BETA, BETA_FAC, PLATO_TOL)
beta_KL = Beta_Scheduler_Plato(beta.beta, 1 / BETA_FAC, PLATO_TOL, ref = beta, threshold=1e-4)


#Данная функция будет выполнятся после каждого шага тренера, 
#соответсвенно нам требуется сделать шаг планировщика и изменить соотвествующий коэффициент
def post_train_step(trainer: VarTrainerParams, train_result: VarBayesTrainer.TrainResult):
    beta.step(train_result.fit_loss)
    beta_KL.step(train_result.dist_loss)
    trainer.params.beta = float(beta)
```
Инициализируем обучающий и валидайионный dataset
```python
#print(model.base_module.state_dict().keys())
val_size    = int(VAL_PERCENT * len(train_dataset)) 
train_size  = len(train_dataset) - val_size 

t_dataset, v_dataset = torch.utils.data.random_split(train_dataset,  
                                                        [train_size,  
                                                            val_size]) 

#Create DataLoaders for the training and validation sets 
train_loader = torch.utils.data.DataLoader(t_dataset,  
                                        batch_size=BATCH_SIZE,  
                                        shuffle=True, 
                                        pin_memory=True) 
eval_loader = torch.utils.data.DataLoader(v_dataset,  
                                        batch_size=BATCH_SIZE,  
                                        shuffle=False, 
                                        pin_memory=True) 
```
К байесовским моделям спокойно применяются все методы nn.Module, в том числе их можно довольно просто перенести на другой device
```python
model.to(device) 
```

После того как мы создали байесовскую сеть, определели loss и задали dataset можно приступить к обучению, с помощью встроенного тренера.
```python
#Если хотим сделать бету фиксированной, то нунжо убрать аргумент [post_train_step]
#trainer = VarBayesTrainer(train_params, ReportChain([VarBaseReport()]), train_loader, eval_loader, [post_train_step])
trainer = VarBayesTrainer(train_params, ReportChain([VarBaseReport()]), train_loader, eval_loader)
trainer.train(model)

```

Также эти модели можно целиком сохранять на диск                  
```python
torch.save(model.state_dict(), 'model_bayes.pt' )
```

И загружать с диска
```python
model.load_state_dict(torch.load('model_bayes.pt'))
image1, label1 = test_dataset[10]
image2, label2 = test_dataset[11]
model(image1)
```

Также с помощью функции eval() тренера можно оценить модель на валидационном dataset

```python
val_loss = 0.0
val_acc = 0.0
PRUNE = 1.0
test_loader = torch.utils.data.DataLoader(test_dataset,  
                                         batch_size=BATCH_SIZE,  
                                         shuffle=False, 
                                         pin_memory=True) 
kl_loss = LogUniformVarKLLoss()
trainer.params.prune_threshold = PRUNE
test_result = trainer.eval(model, test_loader)
acc = test_result.custom_losses['val_accuracy']
print(f'Loss:{test_result.val_loss}, KL Loss: {test_result.dist_loss}, FitLoss: {test_result.fit_loss}, Accuracy {acc}, Prune parameters: {test_result.cnt_prune_parameters}/{test_result.cnt_params}')
```

## Прунинг
Для прунинга моделей рекомендуется использвать какую-то детерминированную оценку моделей. 
В примере сначала проводится прунинг по значению -1.0, а потом устанавливается MAP оценка параметров.
```python
model.to(device=device)
model.prune({'threshold': -1.0})
model.set_map_params()

```

Далее эту модель можно использовать как детерминирвоанную

```python
image, label = test_dataset[100]
plt.imshow(image.permute(1, 2, 0), cmap="gray")
print("Label:", label)
```

    Label: 5



    
![png](reference_files/main_55_1.png)
    



```python
torch.max(model(image.cuda()).data, 1)
```




    torch.return_types.max(
    values=tensor([2.1405], device='cuda:0'),
    indices=tensor([5], device='cuda:0'))


