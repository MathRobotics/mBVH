import numpy as np
import mathrobo as mr
from mbvh import *

def test_init_default():
  node_list = []    
  frame_num = 3
  s_time = 0.001
  frames = np.array(((0,0,0,0,0,0),(1,1,1,1,1,1),(3,3,3,3,3,3)))

  channels = ['Xposition', 'Yposition', 'Zposition']
  parent = BvhNode('test_p', 'JOINT', 0, 0, np.array((0., 0., 0.)), channels)
  node_list.append(parent)   
  child = BvhNode('test_c', 'JOINT', 1, 3, np.array((0., 0., 0.)), channels, parent)
  node_list.append(child)   

  bvh = Bvh(node_list, frame_num, s_time, frames)

  assert bvh.node_list == node_list
  assert bvh.frame_num == frame_num
  assert bvh.sampling_time == s_time
  np.testing.assert_array_equal(bvh.frames, frames)

def test_read_bvh():
  with open('./test.bvh', 'r') as f:
    bvh = Bvh.read_bvh(f)
    
  assert bvh.frame_num == 3
  assert bvh.sampling_time == 0.001
  
  assert bvh.node_num == 6
  
  node = bvh.node_list[0]
  assert node.id == 0
  assert node.dof_index == 0
  assert node.name == 'root'
  assert node.type == 'ROOT'
  np.testing.assert_array_equal(node.offset, np.array((0.,0.,0.)))
  assert node.channels == ['Xposition', 'Yposition', 'Zposition', 'Yrotation', 'Xrotation', 'Zrotation']
  assert node.parent == None
  
  node = bvh.node_list[1]
  assert node.id == 1
  assert node.dof_index == 6
  assert node.name == 'joint1'
  assert node.type == 'JOINT'
  np.testing.assert_array_equal(node.offset, np.array((0.,1.,0.)))
  assert node.channels == ['Yrotation', 'Xrotation', 'Zrotation']
  assert node.parent == bvh.node_list[0]
 
  node = bvh.node_list[2]
  assert node.id == 2
  assert node.dof_index == 9
  assert node.name == 'joint2'
  assert node.type == 'JOINT'
  np.testing.assert_array_equal(node.offset, np.array((0.,1.,0.)))
  assert node.channels == ['Yrotation', 'Xrotation', 'Zrotation']
  assert node.parent == bvh.node_list[1]
  
  node = bvh.node_list[3]
  assert node.id == 3
  assert node.dof_index == 12
  assert node.name == 'Site'
  assert node.type == 'End'
  np.testing.assert_array_equal(node.offset, np.array((0.,5.,0.)))
  assert node.channels == []
  assert node.parent == bvh.node_list[2]
  
  node = bvh.node_list[4]
  assert node.id == 4
  assert node.dof_index == 12
  assert node.name == 'joint3'
  assert node.type == 'JOINT'
  np.testing.assert_array_equal(node.offset, np.array((0.,1.,0.)))
  assert node.channels == ['Yrotation', 'Xrotation', 'Zrotation']
  assert node.parent == bvh.node_list[1]
  
  node = bvh.node_list[5]
  assert node.id == 5
  assert node.dof_index == 15
  assert node.name == 'Site'
  assert node.type == 'End'
  np.testing.assert_array_equal(node.offset, np.array((0.,5.,0.)))
  assert node.channels == []
  assert node.parent == bvh.node_list[4]
  
  assert bvh.sampling_time == 0.001
  assert bvh.frame_num == 3
  