import numpy as np
import matplotlib.pyplot as plt

from mbvh import *

def node_kinematics(bvh, node, p_frame, frame_index, frame_list):
  rel_frame = bvh.calc_relative_frame(node, frame_index)
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

def plot_frame(bvh, frame_index):
  frame_list = kinematics(bvh, frame_index)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for node in bvh.node_list:
    # print(frame_list[node.id])
    if node.parent:
        p_id = node.parent.id
        ax.plot(
            [frame_list[node.id][0, 3], frame_list[p_id][0, 3]],
            [frame_list[node.id][1, 3], frame_list[p_id][1, 3]],
            [frame_list[node.id][2, 3], frame_list[p_id][2, 3]],
            color="black"
        )
  
  # 軸の設定
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([0, 3])
  ax.set_zlim([0, 3])
  
  # 各行列を描画
  for i, frame in enumerate(frame_list):
    origin = frame[:3, 3]  # 平行移動部分 (x, y, z)
    x_axis = frame[:3, 0]  # x軸
    y_axis = frame[:3, 1]  # y軸
    z_axis = frame[:3, 2]  # z軸
    
    # 原点を描画
    ax.scatter(*origin, label=f'Frame {i}')
    
    # 軸を描画
    ax.quiver(*origin, *x_axis, color='r', length=0.1, normalize=True, label=f'X{i}')
    ax.quiver(*origin, *y_axis, color='g', length=0.1, normalize=True, label=f'Y{i}')
    ax.quiver(*origin, *z_axis, color='b', length=0.1, normalize=True, label=f'Z{i}')
  
  # plt.legend()
  plt.show()

def main():
  with open('./test.bvh', 'r') as file:
    file_content = file.read()
  bvh = Bvh.parse_bvh(file_content)

  bvh.show_node_tree()
  bvh.show_frame(0)

  plot_frame(bvh, 0)

if __name__ == "__main__":
    main()