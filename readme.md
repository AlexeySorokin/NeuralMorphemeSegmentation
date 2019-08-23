![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

### English version

Code for articles [Deep Convolutional Networks for Supervised Morpheme Segmentation of Russian Language](https://link.springer.com/chapter/10.1007/978-3-030-01204-5_1) and [Convolutional neural networks for low-resource morpheme segmentation](https://www.aclweb.org/anthology/W19-4218).

To train a model run, for example, ```python neural_morph_segm.py config/sigmorphon/nahuatl_basic.json```. This will train a basic convolutional model for Nahuatl data from the latter paper. To train a one-sided convolutional model, do ```python neural_morph_segm.py config/sigmorphon/nahuatl_lr.json``` and use ```python neural_morph_segm.py config/sigmorphon/nahuatl_lm_new.json``` to train unsupervised model from the same article.

Check the fields in the `.json` config files to specify data, model and results paths. To load a model without training it change `save_file` to `load_file` in the configuration file and add `to_train = False` record to it.

### Русская версия (не обновлялась)

# Разбиение на морфемы с помощью нейронных сетей

Репозиторий содержит нейронную модель для автоматического деления на морфемы, 
обученную на подвыборке, взятой из морфологического словаря А. Н. Тихонова. 
Статья с описанием модели содержится [по ссылке](Articles/MorphemeSegmentation_final.pdf).

Чтобы протестировать обученную модель на тестовой подвыборке, запустите

```python neural_morph_segm.py config/morph_config_load.json```

Чтобы повторить обучение модели, запустите

```python neural_morph_segm.py config/morph_config.json```

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