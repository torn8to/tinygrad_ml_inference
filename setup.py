from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tinygrad_ml_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', ]),
        (os.path.join('share/' + package_name, 'models'), glob("models/*")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nathan',
    maintainer_email='nathanroger314@gmail.com',
    description='A Packge for python based ML inference in tiny grad for depth and segmentation masks',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "tinygrad_ml_inference.depth_node:main"
        ],
    },
)
