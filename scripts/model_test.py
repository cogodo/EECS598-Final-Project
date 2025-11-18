from src.model import TinyLlama

model = TinyLlama()  # Uses default config path
model.load_model()
result = model.generate("Hello, what is your name?")
print(result)