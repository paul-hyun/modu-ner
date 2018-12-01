#nsml: nsml/default_ml:cuda9_torch1.0
from distutils.core import setup

setup(
    name='NER_NSML_Baseline',
    version='1',
    description='NER_NSML_Baseline',
    install_requires=[
        'tensorflow_hub',
        'datetime'
    ]
)
