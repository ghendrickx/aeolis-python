
from aeolis.console_debug import aeolis_debug
import cProfile

def main()-> None:
    '''Runs AeoLiS model in debugging mode. Run this script to start AeoLiS with debugging features enabled, 
    such as step-by-step execution and detailed logging. Useful for development and troubleshooting.
    '''

    configfile = r'c:\Users\aeolis.txt' # Path to the configuration file
    aeolis_debug(configfile)

if __name__ == '__main__':
    main()
