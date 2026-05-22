from setuptools import setup, find_packages

setup(
    name="forestguard",
    version="1.0.0",
    description="Real-time deforestation detection via SAR + optical fusion",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "fg-preprocess=scripts.preprocess:main",
            "fg-train=scripts.train:main",
            "fg-infer=scripts.infer:main",
        ]
    },
)
