[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — [Название проекта]

**Студент:** [Липчанская Софья Игоревна]

**Группа:** [БИВ238]


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

По характеристикам заказа (тип доставки, регион, категория товара, способ оплаты, размер скидки и др.) необходимо предсказать, будет ли доставка задержана.

**Задача:** [Бинарная классификация]

**Датасет:** [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) — 180 519 строк, 53 колонки

**Целевая метрика:** [F1-score (дополнительно: Accuracy, ROC-AUC)]


## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходный датасет DataCo
├── models                      # Сохранённые модели
├── notebooks
│   └── eda.ipynb               # EDA + baseline + эксперименты
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-yooooy811-tech.git
cd hseml-group-project-yooooy811-tech

# 2. Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Скачать датасет с Kaggle и положить в data/raw/

# 5. Запустить ноутбук
jupyter notebook notebooks/eda.ipynb
```

## Данные
- `data/raw/` — исходный файл `DataCoSupplyChainDataset.csv`
- `data/processed/` — предобработанные данные


## Результаты

| Модель | Accuracy | F1-score | ROC-AUC | Примечание |
|--------|----------|----------|---------|------------|
| Logistic Regression (baseline) | 0.7206 | 0.6875 | 0.7377 | |
| Decision Tree | 0.7327 | 0.7111 | 0.7468 | max_depth=10 |
| Random Forest | 0.7569 | 0.7503 | 0.7666 | n_estimators=100 |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
