"""
Created on Tue Jul 14 18:32:47 2020

@author: Shidiq Nur Hidayat
"""

from setuptools import setup

setup(name='nanolib',
      version='0.0.19',
      description='The Data Science Assistant from NANOSENSE',
      url='https://github.com/nanosenseinstrument/nanolib.git',
      author='Shidiq Nur Hidayat',
      author_email='s.hidayat@nanosense-id.com',
      license='MIT',
      packages=['nanolib'],
      install_requires=['scikit-learn', 'numpy', 'seaborn', 'matplotlib', 'sklearn-genetic', 'pycm', 'pandas'],
      zip_safe=False)
