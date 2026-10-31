# TrainPilot — Architecture

```
trainpilot/
├── trainpilot/
│   ├── __init__.py       # публичный API: Pilot
│   ├── measure.py        # замер шагов, памяти, ожидания данных
│   ├── diagnose.py       # правила: «ждёт данные» / «занят»
│   ├── experiments.py    # прогон с AMP (и опционально torch.compile)
│   ├── report.py         # отчёт + готовый кусок кода
│   └── cli.py            # trainpilot run train.py
├── examples/
│   ├── 01_cifar_data_bound.py
│   ├── 02_tinystories_fp32.py
│   └── 03_resnet50_compute_bound.py
├── notebooks/
│   └── trainpilot_colab.ipynb
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Как подключается (идея API)
```python
from trainpilot import Pilot
pilot = Pilot(model, optimizer, loss_fn, loader)
pilot.run()          # замер + эксперимент + отчёт
```
Альтернатива: обёртка шага
```python
for x, y in pilot.wrap(loader):   # мерит ожидание данных
    with pilot.step():            # мерит вычисления
        ...
```

## Поток данных
1. `measure` → сырые замеры (время шагов, ожидание, память)
2. `diagnose` → причина + уверенность
3. `experiments` → базовый vs AMP, проверка loss/NaN
4. `report` → markdown + готовый код

## Ключевые правила замера
- Всегда `torch.cuda.synchronize()` перед засечкой времени
- Прогрев не считается
- Медиана по шагам
- Одинаковый seed и одинаковые батчи для сравнения
