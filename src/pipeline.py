from src.stage.segmentation import run_segmentation
from src.stage.generation import run_generation

def run_pipeline():
    run_segmentation()
    print("Segmentation stage finished successfully.")

    run_generation()
    print("Generation stage finished successfully.")