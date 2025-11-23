from src.model import TinyLlama

model = TinyLlama()  # Uses default config path
model.load_model()
result = model.generate("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
print(result)