# authors_name = 'Preetham Ganesh'
# project_title = 'Digit Recognizer'
# email = 'preetham.ganesh2021@gmail.com'


from utils import create_log
from utils import log_information


def main():
    log_information('')


if __name__ == '__main__':
    major_version = 1
    minor_version = 0
    revision = 0
    global version
    version = '{}.{}.{}'.format(major_version, minor_version, revision)
    create_log('logs', 'model_training_testing_v{}.log'.format(version))
    main()
