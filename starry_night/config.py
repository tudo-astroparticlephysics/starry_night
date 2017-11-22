import os
import pkg_resources
import logging
from configparser import RawConfigParser
from datetime import datetime
from dateutil.parser import parse as parse_date

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


def get_config_for_timestamp(configs, image_timestamp):
    date_sorted_configs = sorted(configs, key=lambda c: c['properties']['useConfAfter'])

    best_config_date = datetime.min
    best_config = None

    for config in date_sorted_configs:
        config_date = parse_date(config['properties']['useConfAfter'])
        if config_date < image_timestamp and config_date > best_config_date:
            best_config = config

    if best_config is None:
        raise ValueError('No config found for {}'.format(image_timestamp))

    return best_config
