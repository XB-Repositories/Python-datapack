from setuptools import setup

setup(
   name='datapack',
   version='0.0.1',
   author='Xabier Benavides',
   author_email='xabier.benavides@ehu.eus',
   packages=['datapack', 'datapack.test'],
   url='',
   license='LICENSE.txt',
   description='Esta librería incluye clases y funciones que pueden utilizarse para trabajar con conjuntos de datos.',
   long_description=open('README.txt').read(),
   python_requires='>=3.8.10',
   tests_require=['pytest'],
   install_requires=[
      "seaborn >= 0.11.0",
      "pandas >= 1.3.2",
      "matplotlib >= 3.4.3",
      "numpy >= 1.19.0",
      "ipython >= 7.31.0"
   ],
)
