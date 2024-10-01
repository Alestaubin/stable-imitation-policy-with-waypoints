from setuptools import setup, find_packages

setup(
    name='stable-imitation-policy-with-waypoints',  # Replace with your package name
    version='0.1.0',  # Initial version
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        # Add package dependencies here
        # 'numpy>=1.18.5',
        # 'requests>=2.23.0',
    ],
    author='Alexandre St-Aubin',
    author_email='alexandre.st-aubin2@mail.mcgill.ca',
    description='Learning globally stable dynamical systems policies through imitation. A modification of the original work, focussing on waypoint-based imitation learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Alestaubin/stable-imitation-policy-with-waypoints',  # Your package's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Python version compatibility
)
