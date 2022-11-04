# authors_name = 'Preetham Ganesh'
# project_title = 'Digit Recognizer'
# email = 'preetham.ganesh2021@gmail.com'


import argparse


def main(args):
    print()


if __name__ == "__main__":
    # Parses arguments.
    parser = argparse.ArgumentParser(description='Digit Recognizer')
    parser.add_argument('-mc', '--model_configuration', default=None, type=str, required=True)
    args = parser.parse_args()
    main(args)
