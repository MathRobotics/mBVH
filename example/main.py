import numpy as np

from mbvh import *

def node_kinematics(bvh, node, p_frame, frame_index, frame_list):
  rel_frame = bvh.calc_joint_rel_frame(node, frame_index)
  frame = p_frame @ rel_frame
  frame_list.append(frame)

  if(node.children):
    for child in node.children:
      node_kinematics(bvh, child, frame, frame_index, frame_list)
      
def kinematics(bvh, frame_index):
  frame_list = []
  frame = np.identity(4)
  node_kinematics(bvh, bvh.node_list[0], frame, frame_index, frame_list)
  
  return frame_list

def main():
  with open('./test.bvh', 'r') as f:
    bvh = Bvh.read_bvh(f)
      
  frame_list = kinematics(bvh, 0)
  print(frame_list)

if __name__ == "__main__":
    main()