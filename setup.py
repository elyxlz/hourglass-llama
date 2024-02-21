from setuptools import setup, find_packages

setup(
    name="hourglass-llama",
    version="0.0.1",
    author="Elio Pascarelli",
    author_email="elio@pascarelli.com",
    description="A clean, modern, and generalized implementation of causal hierarchical transformers.",
    long_description=open('README.md').read(),
    long_description_content_type = 'text/markdown',
    url="https://github.com/elyxlz/hourglass-llama",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "python-dotenv",
    ],
    extras_require={"train": ["accelerate", "wandb", "tqdm"]},
    license="MIT",
)
