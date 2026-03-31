import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="patatas",
    version="0.1",
    author="colindrouineau",
    author_email="colin.drouineau@etu.minesparis.psl.eu",
    packages=["algorithms", "data_process_and_analysis"],
    description="early blight detection using ML algorithms.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/colindrouineau/DIMA_patatas",
    license='MIT',
    python_requires='>=3.12',
    install_requires=[]
)