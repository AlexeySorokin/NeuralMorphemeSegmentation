![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# Разбиение на морфемы с помощью нейронных сетей

Репозиторий содержит нейронную модель для автоматического деления на морфемы, 
обученную на подвыборке, взятой из морфологического словаря А. Н. Тихонова. 
Статья с описанием модели содержится [по ссылке](Articles/MorphemeSegmentation_final.pdf).

Чтобы протестировать обученную модель на тестовой подвыборке, запустите

```python neural_morph_segm.py config/morph_config_load.py```

Чтобы повторить обучение модели, запустите

```python neural_morph_segm.py config/morph_config.py```

## Структура репозитория

* [Articles](Articles): статьи, использующие код из реепозитория.
* [сonfig](config): конфигурационные файлы.
* [data](data): обучающая и контрольная выборка (случайное разбиение морфологического словаря А. Н. Тихонова).
* [models](models): сохранённые модели.
* [neural_morph_segm.py](neural_morph_segm.py): основной файл с кодом модели.
* [read.py](read.py): чтение входных данных.
* [tabled_trie.py](tabled_trie.py): вспомогательные манипуляции с префиксным бором.

## Зависимости

* python3
* keras
* theano или tensorflow
* numpy