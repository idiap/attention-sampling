#!/usr/bin/env python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse
import os
from os import path
from subprocess import call
import tempfile

import yaml


SCRIPT_TPL = """#!/bin/bash
git clone . {dir}/project
{commands}
"""

RESERVED_NAMES = [
    "image",
    "services",
    "stages",
    "types",
    "before_script",
    "after_script",
    "variables",
    "cache"
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Approximate running the scripts from the .gitlab-ci.yml"
    )
    parser.add_argument(
        "--build_dir",
        help="Set a build directory"
    )

    args = parser.parse_args()

    # Make the temporary build directory
    build_dir = tempfile.mkdtemp(dir=args.build_dir)
    print("Building in", build_dir)

    # Collect the commands from the yaml file
    commands = []
    pipeline = yaml.load(open(".gitlab-ci.yml"))
    for stage in pipeline["stages"]:
        for k in pipeline:
            if k in RESERVED_NAMES:
                continue
            job = pipeline[k]
            if not job.get("stage", "test") == stage:
                continue
            commands.append("cd {}/project".format(build_dir))
            commands.extend(job["script"])

    # Build the script
    script = SCRIPT_TPL.format(
        dir=build_dir,
        commands="\n".join(commands)
    )
    handle, script_file = tempfile.mkstemp(dir=build_dir)
    os.close(handle)
    with open(script_file, "w") as f:
        f.write(script)

    # Execute it by replacing this process with bash
    call(["/bin/bash", script_file])
