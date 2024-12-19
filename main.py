import cherrypy
import numpy as np
from tensorflow import keras

class IrisClassifier:
    @cherrypy.expose
    def index(self):
        return """
        <html>
        <head><title>Классификация ирисов</title></head>
        <body>
            <h1>Введите параметры цветка ирис</h1>
            <form method="post" action="predict">
                Длина чашелистика: <input type="text" name="sepal_length"><br>
                Ширина чашелистика: <input type="text" name="sepal_width"><br>
                Длина лепестка: <input type="text" name="petal_length"><br>
                Ширина лепестка: <input type="text" name="petal_width"><br>
                <button type="submit">Отправить</button>
            </form>
        </body>
        </html>
        """

    @cherrypy.expose
    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        try:
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)
        except ValueError:
             return "Ошибка: Введите числовые значения для параметров."

        try:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = self.model.predict(input_data)
            predicted_class_index = np.argmax(prediction)
            predicted_class = self.iris_classes[predicted_class_index]
            return f"Предсказанный класс: {predicted_class}"
        except Exception as e:
            return f"Ошибка предсказания: {e}"


    def __init__(self):
        try:
            self.model = keras.models.load_model("iris_model.keras")
            print("Модель загружена успешно!")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None

        self.iris_classes = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}



if __name__ == '__main__':
    cherrypy.quickstart(IrisClassifier())
