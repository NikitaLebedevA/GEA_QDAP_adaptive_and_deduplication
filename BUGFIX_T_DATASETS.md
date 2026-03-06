# Bugfix: зависание на T-датасетах

## Симптом

При запуске на T-датасетах (T1–T8, J=100–1600) алгоритм зависает на этапе инициализации популяции. Heartbeat показывает 0/600 задач завершено даже спустя 1.5+ часа. На c-датасетах (J=20–35) всё работает нормально.

## Корневая причина

Функция `heuristic2` (инициализация первого индивида популяции) содержала **слабый repair-механизм**, который не соответствовал MATLAB-оригиналу. Из-за этого для T-датасетов с жёсткими ограничениями ёмкости `heuristic2` возвращала **нефизибельное** решение (cost = inf).

Далее цикл инициализации популяции мутировал это нефизибельное решение в надежде получить физибельное:

```python
while len(population) < cfg.population_size:
    mutated = mutation(population[0].permutation, model, rng)
    individual = evaluate_permutation(mutated, model)
    if math.isfinite(individual.cost):
        population.append(individual)
```

Мутации нефизибельного базового решения почти никогда не дают физибельного потомка → бесконечный цикл → все 16 воркеров заблокированы → 0 задач завершено.

## Почему MATLAB не зависал

MATLAB-код имеет тот же `while Cost == inf` цикл без таймаута, но его `Heuristic2.m` содержит **каскадный repair** с внешним `while min(cvar) < 0`:

```matlab
% MATLAB Heuristic2.m, строки 50-70
while min(cvar) < 0                    % повторять пока ВСЕ не станут физибельными
    for i = 1:I
        while cvar(i) < 0
            [a, b] = max(Wij(i,:));    % тяжелейший job на перегруженном facility
            ... удаляем ...
            [c, d] = min(aij(:,b));    % перемещаем на facility с мин. весом
            if d == i
                [c, d] = max(cvar);    % или на facility с макс. запасом
            end
            ... назначаем БЕЗ проверки ёмкости ...  % каскадный ремонт!
        end
    end
end
```

Ключевые отличия MATLAB от старого Python:

| Аспект | MATLAB | Python (было) |
|--------|--------|---------------|
| Количество проходов | Внешний `while` — повторяет до полной физибельности | Один проход `for i in range(I)` |
| Куда перемещает job | На facility с мин. весом **или** макс. запасом, **без проверки** ёмкости | **Только** на facility с гарантированной ёмкостью |
| Если нет места | Перемещает всё равно — каскад починит | **Сдаётся**: возвращает job обратно и делает `break` |

## Что было исправлено

### 1. Repair в `heuristic2` (основной фикс)

**Файлы:**
- `gea_gqap_adaptive_python/gea_gqap_adaptive_python/heuristics.py`
- `GEA_GQAP_Python/gea_gqap_python/heuristics.py`

**Было (слабый repair):**
```python
for i in range(I):
    while count[i] > model.bi[i] + 1e-9:
        ...
        for new_i in target:
            if count[new_i] + model.aij[new_i, job] <= model.bi[new_i]:
                ... assign ...
                break
        else:
            X[i, job] = 1       # возвращает обратно
            break                # сдаётся
```

**Стало (MATLAB-style каскадный repair):**
```python
cvar = model.bi - count
Wij = X * model.aij
max_repair_passes = I * J
repair_pass = 0
while np.any(cvar < -1e-9) and repair_pass < max_repair_passes:
    repair_pass += 1
    for i in range(I):
        while cvar[i] < -1e-9:
            assigned_jobs = np.where(X[i] == 1)[0]
            if assigned_jobs.size == 0:
                break
            b = assigned_jobs[np.argmax(Wij[i, assigned_jobs])]
            count[i] -= model.aij[i, b]
            cvar[i] = model.bi[i] - count[i]
            X[i, b] = 0
            Wij[i, b] = 0
            d = int(np.argmin(model.aij[:, b]))
            if d == i:
                d = int(np.argmax(cvar))
            count[d] += model.aij[d, b]
            cvar[d] = model.bi[d] - count[d]
            X[d, b] = 1
            Wij[d, b] = model.aij[d, b]
```

Теперь `heuristic2` гарантированно возвращает физибельное решение, как в MATLAB. Добавлена страховка `max_repair_passes = I * J` от теоретически возможного бесконечного цикла.

### 2. Time-limit на инициализацию популяции (страховка)

**Файлы:**
- `gea_gqap_adaptive_python/gea_gqap_adaptive_python/algorithm.py`
- `gea_gqap_adaptive_python/gea_gqap_adaptive_python/algorithm_adaptive.py`
- `GEA_GQAP_Python/gea_gqap_python/algorithm.py`

**Было:**
```python
while len(population) < cfg.population_size:
    mutated = mutation(population[0].permutation, model, rng)
    individual = evaluate_permutation(mutated, model)
    if math.isfinite(individual.cost):
        population.append(individual)
```

**Стало:**
```python
init_budget = (cfg.time_limit * 0.1) if cfg.time_limit is not None else None
while len(population) < cfg.population_size:
    if init_budget is not None and (time.perf_counter() - start_time) >= init_budget:
        break
    mutated = mutation(population[0].permutation, model, rng)
    individual = evaluate_permutation(mutated, model)
    if math.isfinite(individual.cost):
        population.append(individual)
```

Инициализация популяции теперь ограничена 10% от `time_limit` (100 секунд при лимите 1000с). Если за это время не удалось набрать `population_size`, алгоритм работает с тем, что есть.

### 3. Time-limit на дозаполнение в `_select_population_dedupe` (страховка)

**Файлы:**
- `gea_gqap_adaptive_python/gea_gqap_adaptive_python/algorithm_adaptive.py`
- `GEA_GQAP_Python/gea_gqap_python/algorithm.py`

Сигнатура функции расширена:
```python
def _select_population_dedupe(
    pool, population_size, model, rng,
    start_time=0.0, time_limit=None,  # новые параметры
)
```

Цикл дозаполнения теперь прерывается при исчерпании `time_limit`:
```python
while len(unique_list) < population_size:
    if time_limit is not None and (time.perf_counter() - start_time) >= time_limit:
        break
    ...
```

Вызовы обновлены с передачей `start_time` и `cfg.time_limit`.

## Почему c-датасеты не были затронуты

c-датасеты (J=20–35, I=15–95) имеют достаточно свободные ограничения ёмкости. Даже слабый repair почти всегда находил facility с гарантированной ёмкостью для перемещения, поэтому `heuristic2` возвращала физибельное решение. T-датасеты (J=100–1600) с жёсткими ограничениями — нет.
