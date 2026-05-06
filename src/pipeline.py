from src.stage.segmentation import run_segmentation
from src.stage.generation import run_generation
from src.stage.placement import run_placement

def run_pipeline():
    run_segmentation()
    print("Segmentation stage finished successfully.")

    run_generation()
    print("Generation stage finished successfully.")

    run_placement()
    print("Placement stage finished successfully.")