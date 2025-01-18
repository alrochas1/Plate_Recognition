import argparse
import tensorflow as tf
from neural_model.neural_network_train import training
from neural_model.neural_network_test import testing


def parse_args():
    parser = argparse.ArgumentParser(description="Seleccione el modo de operación: train/test")
    parser.add_argument('mode', choices=['train', 'test'], default='test', help="Modo de operación: 'train' o 'test'.")
    return parser.parse_args()

# Función principal
def main():
    args = parse_args()
    
    if args.mode == 'train':
        print("Iniciando el entrenamiento del modelo...")
        training()
    elif args.mode == 'test':
        print("Iniciando las pruebas del modelo...")
        testing()

if __name__ == "__main__":
    main()



