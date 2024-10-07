import numpy as np
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
  
  assert bvh.end_list[0].id == 3
  assert bvh.end_list[1].id == 5
  
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

  np.testing.assert_allclose(bvh.frames[0], np.array((0,0,0,0,0,0,0,0,0.1,0,0,0.1,0,0,0.1)))
  np.testing.assert_allclose(bvh.frames[1], np.array((0,0,0,0,0,0,0,0,0.2,0,0,0.2,0,0,0.2)))
  np.testing.assert_allclose(bvh.frames[2], np.array((0,0,0,0,0,0,0,0,0.3,0,0,0.3,0,0,0.3)))
  
def test_get_joint():
  with open('./test.bvh', 'r') as f:
    bvh = Bvh.read_bvh(f)

  assert bvh.get_joint('joint1') == bvh.node_list[1]
  assert bvh.get_joint('joint2') == bvh.node_list[2]
  assert bvh.get_joint('joint3') == bvh.node_list[4]
  
  warnings.simplefilter('ignore')
  assert bvh.get_joint('root') == None
  assert bvh.get_joint('Site') == None

def test_get_node_id():
  with open('./test.bvh', 'r') as f:
    bvh = Bvh.read_bvh(f)

  assert bvh.get_node_id('root') == 0
  assert bvh.get_node_id('joint1') == 1
  assert bvh.get_node_id('joint2') == 2
  assert bvh.get_node_id('joint3') == 4
  
  warnings.simplefilter('ignore')
  assert bvh.get_node_id('aaa') == 0

def test_get_node_dof_index():
  with open('./test.bvh', 'r') as f:
    bvh = Bvh.read_bvh(f)

  assert bvh.get_node_dof_index('root') == 0
  assert bvh.get_node_dof_index('joint1') == 6
  assert bvh.get_node_dof_index('joint2') == 9
  assert bvh.get_node_dof_index('joint3') == 12
  
  warnings.simplefilter('ignore')
  assert bvh.get_node_dof_index('aaa') == 0
  
def test_get_node_frame():
  with open('./test.bvh', 'r') as f:
    bvh = Bvh.read_bvh(f)
    
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[0], 0),np.array((0,0,0,0,0,0)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[1], 0),np.array((0,0,0.1)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[2], 0),np.array((0,0,0.1)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[3], 0),np.array([], dtype='float32'))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[4], 0),np.array((0,0,0.1)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[5], 0),np.array([], dtype='float32'))
  
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[0], 1),np.array((0,0,0,0,0,0)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[1], 1),np.array((0,0,0.2)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[2], 1),np.array((0,0,0.2)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[3], 1),np.array([], dtype='float32'))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[4], 1),np.array((0,0,0.2)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[5], 1),np.array([], dtype='float32'))

  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[0], 2),np.array((0,0,0,0,0,0)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[1], 2),np.array((0,0,0.3)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[2], 2),np.array((0,0,0.3)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[3], 2),np.array([], dtype='float32'))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[4], 2),np.array((0,0,0.3)))
  np.testing.assert_allclose(bvh.get_node_frame(bvh.node_list[5], 2),np.array([], dtype='float32'))