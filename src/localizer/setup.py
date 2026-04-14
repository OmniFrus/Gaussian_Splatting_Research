from setuptools import find_packages, setup

package_name = 'localizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='imp-a2',
    maintainer_email='l.hidding.1@student.rug.nl',
    description='Localizer uses an RGBD camera with built in IMU to localize and object in both space and time.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        	'save_images = localizer.camera_subscriber:main',
        	'simulate = localizer.camera_simulator:main',
        	'config_ui = localizer.config_ui:main',
        ],
    },
)
