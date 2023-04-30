"""
Usage:
    channeltransformer_main.py --config=config_name [--gpu=<gpu>]
    
Options:
    --gpu=<gpu>     Choice of GPU [default:0]
"""

from pytorch.configs.experiment_config import *
from docopt import docopt
if __name__ == '__main__':
    args = docopt(__doc__)
    launcher_cls, model, trainer, data, options = eval(args['--config'])()
    options['trainer_options']['gpu'] = int(args['--gpu'])
    launcher = launcher_cls(options, data, model, trainer)
    launcher.run()
