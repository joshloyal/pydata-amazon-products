from setuptools import setup


PACKAGES = [
    'amazon_product'
]


def setup_package():
    setup(
        name="pydata-amazon-products",
        version='0.1.0',
        description="Code for PyData Talk on 'Classifying Products Based on Images and Text using Keras'."
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/pydata-amazon-products',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn', 'keras'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
