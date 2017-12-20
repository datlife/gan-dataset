
import os
import sys
from jinja2 import Environment, FileSystemLoader


def writeHTML(filename, html_template=None, **kwargs):
    env             = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], 'templates')))
    template        = env.get_template(html_template)

    # every day I am shuffling!..
    parsed_template = template.render(**kwargs)

    # to save the results
    with open(filename, "w") as fh:
        fh.write(parsed_template)


def relative_path(ref_path, target_path):
    common_prefix = os.path.commonprefix([ref_path, target_path])
    return os.path.relpath(target_path, common_prefix)

