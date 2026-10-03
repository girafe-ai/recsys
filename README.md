# Recommender Systems — 26f

## Расписание и материалы

| № | Тип | Тема | Материалы | Запись |
|---:|---|---|---|---|
| 1 | Лекция | Введение в рекомендательные системы | [Слайды](lectures/01_intro/slides.pdf) | — |
| 1 | Семинар | Первые шаги в рекомендательных системах | [Ноутбук](seminars/01_intro/seminar.ipynb) | — |
| 2 | Лекция | Ранжирование и метрики | [Слайды](lectures/02_ranking_and_metrics/slides.pdf) | — |
| 2 | Семинар | Ранжирование | [Ноутбук](seminars/02_ranking/seminar.ipynb) | — |
| 3 | Семинар | Коллаборативная фильтрация: item-to-item и user-to-user | [Ноутбук](seminars/03_cf/seminar.ipynb) | — |
| 4 | Лекция | Матричные факторизации | [Слайды](lectures/04_matrix_factorizations/slides.pdf) | — |
| 4 | Семинар | Матричная факторизация: ALS | [Ноутбук](seminars/04_als/seminar.ipynb), [Drive / Colab](https://drive.google.com/file/d/1YjpSmw1RXlzczJTdZBC3Nev8-F95Td7W/view?usp=drivesdk) | — |

Материалы занятий хранятся в репозитории. Датасеты, результаты экспериментов и кеши в Git не добавляются.

## Окружение

Для семинаров используется изолированное Conda-окружение. После установки Miniconda создайте его из корня репозитория:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"  # если команда conda ещё не доступна
conda env create -f environment.yml
conda activate recsys
```
