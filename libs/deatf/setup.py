from distutils.core import setup
setup(
  name = 'deatf',         # How you named your package folder (MyLib)
  packages = ['deatf'],   # Chose the same as "name"
  version = '1.0',      # Start with a small number and increase it with every change you make
  license='LGPLv3',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Distributed Evolutionary Algorithms in TensorFlow (DEATF) is a framework where networks generated with TensorFlow are evolved via DEAP.',   # Give a short description about your library
  author = 'Ivan Hidalgo',                   # Type in your name
  author_email = 'ivanhcenalmor@domain.com',      # Type in your E-Mail
  url = 'https://github.com/IvanHCenalmor/deatf',   # Provide either the link to your github or to your website
  keywords = ['NEUROEVOLUTION', 'DEAP', 'TENSORFLOW', 'GENETIC', 'ALGORITHMS'],   # Keywords that define your package best
  install_requires=['numpy', 'tensorflow', 'deap', 'tensorflow-datasets', 'scikit-learn', 'sklearn', 'pandas']
)
