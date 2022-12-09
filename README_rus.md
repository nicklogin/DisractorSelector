# DistractorSelector

Исходный код магистерской дипломной работы Н. Логин "Автоматизированная генерация грамматических вопросов с множественным выбором на базе корпусной аннотации"

Обучающий датасет доступен по адресу https://disk.yandex.ru/d/_k_AwobPf0l5bQ

Папка processed_texts (для эксперимента по предсказанию места ошибки - predict_error_places.ipynb) доступна по адресу https://disk.yandex.ru/d/6t48nLkkC3mF6Q

1. Установка

Для использованием программы необходимо склонировать данный репозиторий и установить зависимости

```bash
python -m pip install -r requirements.txt
```

2. Установка через Docker

DistractorSelector можно установить из Dockerhub-репозитория:

```bash
docker pull niklogin/disselector:latest
```

Также можно собрать Docker-образ самостоятельно:

```bash
docker build . -t disselector:latest
```

3. Использование через командную строку

Данная программа принимает на вход СSV-файл с контекстами предложений и выдает файл с предлагаемыми неправильными вариантами ответа на вопросы с множественным выбором.

Входной файл должен содержать следующие столбцы (см. пример - gold_standard/gold_standard_input.csv):

<b>Masked_sentence</b> - Предложение, в котором "целевое" слово для формирования вопроса заменено на специальный токен [MASK]

<b>Right_answer</b> - целевое слово

Запуск программы осуществляется следующей командой
```bash
python -m distractor_generator --args
```

Аргументы командной строки программы

<table>
    <th>
        <tr>
            <td>Аргумент</td>
            <td>Описание</td>
            <td>Значение по умолчанию</td>
        </tr>
    </th>
    <tr>
        <td>--filename</td>
        <td>Путь к входному файлу</td>
        <td>gold_standard/gold_standard_input.csv</td>
    </tr>
    <tr>
        <td>--output_filename</td>
        <td>Путь к выходному файлу</td>
        <td>>data/gold_standard_output.csv</td>
    </tr>
    <tr>
        <td>--sep</td>
        <td>Разделитель полей в CSV-файле</td>
        <td>;</td>
    </tr>
    <tr>
        <td>--index_col</td>
        <td>Название столбца-индекса в CSV-файле</td>
        <td>Без названия</td>
    </tr>
    <tr>
        <td>--n</td>
        <td>Количество вариантов, поступающих на вход классификатора</td>
        <td>20</td>
    </tr>
    <tr>
        <td>--no-clf</td>
        <td>Не использовать классификатор</td>
        <td> - </td>
    </tr>
    <tr>
        <td>--clf_path</td>
        <td>Путь к файлу с сохранённой моделью классификатора</td>
        <td>XGBAllFeats/clf.pkl</td>
    </tr>
    <tr>
        <td>--cols_path</td>
        <td>Путь к файлу с перечислением признаков, релевантных для классификатора</td>
        <td>XGBAllFeats/cols.json</td>
    </tr>
</table>

В случае возникновения ошибок на стороне NLTK выполнить следующую команду

```bash
python -m nltk.downloader punkt
```

4. Использование через WebAPI

```bash
python -m api
```

Документация будет доступна по адресу http://localhost:5000/docs
