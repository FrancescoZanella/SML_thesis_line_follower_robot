from robot_controller import RobotController
from pathlib import Path
import sys

def parse_arguments():
    args = {}
    for arg in sys.argv[1:]:
        key, value = arg.split(':')
        args[key.strip()] = value.strip()
    return args
def print_arguments(args):
    for key, value in args.items():
        print(f"{key.upper()}: {value}")
def main():
    args = parse_arguments()
    
    model_path = Path(__file__).parent.parent.parent / 'data' / 'models' / f"{args['model']}.pkl"
    
    controller = RobotController(
        production=args['production'] == 'True',
        model_path=str(model_path),
        plot=args['plot'] == 'True',
        save_sensors=args['save_sensors'] == 'True',
        save_images=args['save_images'] == 'True',
        verbose=args['verbose'] == 'True',
        learning=args['learning'] == 'True',
        enable_recovery=args['enable_recovery'] == 'True'
    )
    print_arguments(args)
    controller.run()

if __name__ == "__main__":
    main()

