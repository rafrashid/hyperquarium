rule all:
    shell:
        """
        echo "No all rule made"
        """

configfile: "workflows/config.yml"

SCAN_RECORDS_PATH = config['scan_records_path']

import itertools

DARK_CURRENT = list(config['dark_current'])

CALIB_SCANS = list(
    itertools.chain(*config['calibration'].values())
)

PILOT_SCANS = list(
    itertools.chain(*config['pilot'].values())
)

OPERATIONAL_SCANS = list(
    itertools.chain(*config['operational'].values())
)

ALL_SCANS = CALIB_SCANS + PILOT_SCANS  #+OPERATIONAL_SCANS

def protect_fields(str, exclude=(), **kwargs):
    class SafeDict(dict):
        def __missing__(self, key):
            if key in exclude:
                return '{' + key + '}'
            else:
                return '{{' + key + '}}'

    replacements = SafeDict(**kwargs)
    return str.format_map(replacements)


import matplotlib

matplotlib.use('agg')
include: "workflows/dark_current_.smk"
include: "workflows/annotations_.smk"
