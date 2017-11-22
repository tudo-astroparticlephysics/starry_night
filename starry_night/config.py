import os
import pkg_resources
import logging
from configparser import RawConfigParser

log = logging.getLogger(__name__)


def load_config(config):

    # give priority to a single file given as argument
    if os.path.isfile(config):
        log.info('Parsing config file: {}'.format(config))
        parser = RawConfigParser()
        parsed_files = parser.read(config)
        if not parsed_files:
            raise IOError('Could not parse config file {}'.format(config))
        return [parser]
    else:
        log.info('config is not a file, falling back to packaged configs')

    configs = list()
    path = pkg_resources.resource_filename('starry_night', 'data/')
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            path = os.path.join(root, f)
            if config in f:
                log.info('Parsing config file: {}'.format(path))

                parser = RawConfigParser()
                try:
                    parsed_files = parser.read(path)
                    if parsed_files:
                        configs.append(parser)
                except UnicodeDecodeError:
                    log.error('Could not open conf file: {}'.format(path))

    if len(configs) == 0:
        log.error('Unable to parse any config file')
        raise

    return configs
