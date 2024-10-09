from setuptools import setup,find_packages
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements
    '''This function will return th list of requirements'''
    
    
setup(
    name="StudentPerformanceIndicator",
    version="0.0.1",
    author="Gaurav",
    author_email="gaurav06704@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)