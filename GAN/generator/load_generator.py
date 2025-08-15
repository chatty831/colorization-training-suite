import torch
from GAN.generator.generator_model import Colorizer, Generator

def load_generator(model_path):
    generator = Generator()
    gen_st_dict = torch.load(model_path)
    generator.load_state_dict(gen_st_dict)
    return generator