# mBVH

## Overview
mBVH is a Python package for manipulating BVH (Biovision Hierarchy) files. This package provides tools to easily load, parse, and manipulate BVH files.

## Installation

### Clone the Repository
Clone the repository to your local machine using:

```bash
git clone https://github.com/MathRobotics/mBVH.git
```

### Install Dependencies
Run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Install the Package
To install mBVH in your local environment, use:

```bash
pip install .
```

## Usage
Here is a simple example of how to use the mBVH package to load a BVH file:

```python
from mbvh import *

with open('path/to/your/file.bvh', 'r') as file:
file_content = file.read()
bvh = Bvh.parse_bvh(file_content)

bvh.show_node_tree()
```

## Testing
To run tests, use the following command:

```bash
pytest
```

## License
This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.