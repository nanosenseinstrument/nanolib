# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:32:47 2020

@author: Shidiq Nur Hidayat
"""

from setuptools import setup

setup(name='nanolib',
      version='0.0.10',
      description='The Data Science Assistant from NANOSENSE',
      url='https://github.com/Shidiq/nanolib.git',
      author='Shidiq Nur Hidayat',
      author_email='s.hidayat@nanosense-id.com',
      license='MIT',
      packages=['nanolib'],
      install_requires=['scikit-learn', 'numpy', 'seaborn', 'matplotlib', 'sklearn-genetic', 'pycm', 'pandas',
                        'tensorflow'],
      zip_safe=False)
