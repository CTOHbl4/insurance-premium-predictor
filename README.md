# Запуск:

В папку generator надо положить папку с двумя файлами .csv (путь для проверки указан в producer.py).

https://www.kaggle.com/datasets/amirjdai/vehicle-insurance-data

Также, в генераторе (producer.py) и .env нужно ip своего компьютера указать (168...).

# Команды запуска (находимся в корне проекта):

Для запуска генератора на свой компьютер нужно скачать pandas и kafka-python>=2.0.2

```
cd model
sudo docker compose up -d --build
sleep 120
cd ../generator
python3 generator/producer.py
```

```
ctrl-c, чтобы прервать producer
cd ../model
docker compose down -v
```
