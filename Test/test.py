from predict import Predictor

predictor = Predictor()
predictor.setup()
caption = predictor.predict(
    image="Images/COCO_val2014_000000060623.jpg",
    model="Oxford",
    use_beam_search=False
)
print(caption)  
caption = predictor.predict(
    image="Images/bokete_0.jpg",
    model="Oxford",
    use_beam_search=False
)
print(caption)
caption = predictor.predict(
    image="Images/bokete_1.jpg",
    model="Oxford",
    use_beam_search=False
)
print(caption)
caption = predictor.predict(
    image="Images/sumo.jpeg",
    model="Oxford",
    use_beam_search=False
)
print(caption)
