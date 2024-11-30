import numpy as np
import matplotlib.pyplot as plt

from mbvh import *

def main():
  with open('./test.bvh', 'r') as file:
    file_content = file.read()
  bvh = Bvh.parse_bvh(file_content)

  bvh.show_node_tree()
  bvh.show_frame(0)

  bvh.plot_frame(0)

if __name__ == "__main__":
    main()