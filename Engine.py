from EngineUpdated.foodDetection.foodCategories.predict import predictFoodCategory
from EngineUpdated.foodDetection.foodDetect import foodDetection
from EngineUpdated.hotelObjectDetection.hotelObjectDetection import hotelObject
from EngineUpdated.landmarkDetection.landmarkDetection import landmark
from EngineUpdated.foodDetection.foodCategories.keras_model import build_model
import os

class Engine:
    def ImagePath(self,imageName):
        image_path = "C:\Tensorflow\models\Research\object_detection\EngineUpdated\images"
        full_pathImage = os.path.join(image_path,imageName)
        return full_pathImage

    def StartEngine(self,full_pathImage):
        foodArray = foodDetection().detectFood(full_pathImage)
        landmarkArray = landmark().DefineLandmark(full_pathImage)
        objectArray = hotelObject().ObjectDetect(full_pathImage)
        bagOfObjects = []
        for food in foodArray:
            bagOfObjects.append(food)
        for land in landmarkArray:
            bagOfObjects.append(land)
        for object in objectArray:
            bagOfObjects.append(object)
        print(bagOfObjects)

obj = Engine().StartEngine(Engine().ImagePath("conf_room1.jpeg"))