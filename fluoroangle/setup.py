from setuptools import setup

setup(
    name='FluoroAngle',
    version='0.1.0',
    description='Calculate gantry angles for fluoro procedures from CT DICOM images',
    url='',
    author='Duncan Ferguson',
    author_email='duncan.ferguson@alumni.ubc.ca',
    license='',
    packages=['fluoroangle'],
    install_requires=['pandas',
                      'numpy',
                      'pydicom',
                      'opencv-python',
                      'scipy',
                      'matplotlib',
                      'tk',
                      'Pillow'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
