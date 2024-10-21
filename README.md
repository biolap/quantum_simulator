# Квантовый симулятор волн

Этот проект реализует 2D-симуляцию поведения волновых пакетов, подчиняющихся уравнению Шрёдингера, с использованием метода ADI (Alternating Direction Implicit).  Симуляция визуализируется в 3D с помощью библиотеки `pyqtgraph`.

## Функциональность:

* **Визуализация волновой функции:**  Отображение вещественной и мнимой частей волновой функции, а также её амплитуды в 3D.
* **Симуляция различных сценариев:**  Возможность запускать симуляцию для различных сценариев: столкновение волновых пакетов, коллапс волновой функции, движение волнового пакета, запутывание.
* **Настройка параметров:**  Настройка параметров симуляции, таких как размер сетки, сглаживание, коллапс волновой функции, и другие.
* **Запись видео:**  Возможность записи видео симуляции.


## Запуск:

1. **Установка зависимостей:** Убедитесь, что у вас установлен Python 3.10 или выше.  Установите необходимые библиотеки с помощью `pip install -r requirements.txt`.
2. **Запуск симулятора:** Запустите скрипт `main.py` с помощью `python main.py`.
3. **Использование GUI:**  Используйте графический интерфейс для настройки параметров и запуска симуляции.


## Структура проекта:


## Технические детали:

* **Язык программирования:** Python 3.10+
* **Библиотеки:**  `numpy`, `scipy`, `matplotlib`, `PyQt6`, `pyqtgraph`, `pyopengl`, `numba`, `Pillow`, `numexpr`
* **Алгоритм:** Метод ADI (Alternating Direction Implicit) для решения уравнения Шрёдингера.


## Авторы:

Dzianis Bialou


## Лицензия:

MIT License


## Будущие разработки:

* Добавление  новых  сценариев  симуляции.
* Улучшение  визуализации.
* Добавление  интерактивного  управления.
