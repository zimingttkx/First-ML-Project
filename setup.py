from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    """
    This function will return a list of the requirements required to run the package.
    :param file_path:
    :return:
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements
setup(
    name="My-First-ML-Project",
    version="0.0.1",
    author="ZiMing",
    author_email="2147514473@qq.com",
    packages= find_packages(),
    install_requires=get_requirements("requirements.txt")

)