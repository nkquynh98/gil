import setuptools

# from distutils.core import setup
# from Cython.Build import cythonize
# setup(
#     ext_modules=cythonize([
#         "pyrieef/geometry/differentiable_geometry.pyx",
#         "pyrieef/geometry/workspace.pyx",
#         "pyrieef/motion/cost_terms.pyx"])
# )

with open("README.md", "r") as fh:
    long_description = fh.read()

namespace_package = 'motion'

setuptools.setup(
    name='gil',
    version='0.0.1',
    author="KhanhQuynhNguyen",
    author_email="nkquynh1998@gmail.com",
    description="Guided Imitation Learning with Mogaze dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nkquynh98/gil",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)