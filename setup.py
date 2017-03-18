from setuptools import setup


PACKAGES = [
    'amazon_products'
]


def setup_package():
    setup(
        name="pydata-amazon-products",
        version='0.1.0',
        description="Code for PyData Talk on 'Classifying Products Based on Images and Text using Keras'.",
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/pydata-amazon-products',
        license='MIT',
        install_requires=['scikit-learn', 'keras', 'scipy', 'numpy'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
