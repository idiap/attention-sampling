#!/usr/bin/env python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Setup attention-sampling"""

from distutils.file_util import copy_file
from multiprocessing import cpu_count
from itertools import dropwhile
import os
from os import path
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from subprocess import CalledProcessError, check_call


class TensorflowExtension(Extension, object):
    @property
    def path(self):
        cmake = [s for s in self.sources if "CMakeLists.txt" in s][0]
        return path.dirname(cmake)

    def compile(self):
        print("Building {}".format(self.name))
        extension_dir = path.join(self.path, "build")
        os.makedirs(extension_dir, exist_ok=True)

        try:
            check_call(
                ["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."],
                cwd=extension_dir
            )
            check_call(
                ["make", "-j{}".format(max(cpu_count()-1, 1))],
                cwd=extension_dir
            )
            check_call(
                ["make", "install"],
                cwd=extension_dir
            )
        except CalledProcessError as e:
            raise RuntimeError(
                "Couldn't compile and install {}".format(self.name)
            ) from e


class custom_build_ext(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, TensorflowExtension):
            ext.compile()
            filename = self.get_ext_filename(ext.name)
            if not self.dry_run:
                os.makedirs(
                    path.join(self.build_lib, path.dirname(filename)),
                    exist_ok=True
                )
            copy_file(filename, path.join(self.build_lib, filename),
                      verbose=self.verbose, dry_run=self.dry_run)
        else:
            super(custom_build_ext, self).build_extension(ext)

    def get_ext_filename(self, fullname):
        return path.sep.join(fullname.split(".")) + ".so"


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("ats", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_extensions():
    return [
        TensorflowExtension(
            "ats.ops.extract_patches.libpatches",
            [
                "ats/ops/extract_patches/extract_patches.h",
                "ats/ops/extract_patches/extract_patches.cc",
                "ats/ops/extract_patches/extract_patches.cu",
                "ats/ops/extract_patches/CMakeLists.txt",
            ]
        )
    ]


def get_install_requirements():
    return [
        "keras>=2",
        "numpy"
    ]


def setup_package():
    with open("README.rst") as f:
        long_description = f.read()
    meta = collect_metadata()
    setup(
        name="attention-sampling",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
        ],
        packages=find_packages(exclude=["docs", "tests", "scripts"]),
        install_requires=get_install_requirements(),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": custom_build_ext}
    )


if __name__ == "__main__":
    setup_package()
