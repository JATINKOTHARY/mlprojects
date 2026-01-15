from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_packages(file_path:str)-> List[str]:
    '''
    Docstring for get_packages
    
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: List[str]

    Returns a list of packages to be installed from the file 
    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements ]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
name = "MLprojects",
version="0.0.1",
author="Jatin",
author_email="jatinkothary123@gmail.com",
packages=find_packages(),
install_requires = get_packages("requirements.txt")
)