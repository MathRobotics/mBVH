import numpy as np
import mathrobo as mr
from mbvh import *

def test_init_default():
  name = 'test_node'
  type_ = 'JOINT'
  id = 11
  dof_index = 20
  node = BvhNode(name, type_, id, dof_index)

  assert node.name == name
  assert node.type == type_
  assert node.id == id
  assert node.dof_index == dof_index
  
def test_init_offset():
  offset = np.array((1., 2., 0.))
  node = BvhNode('test_node', 'JOINT', 1, 0, offset)
  
  np.testing.assert_array_equal(node.offset, offset)
  
def test_init_channels():
  channels = ['Xposition', 'Yposition', 'Zposition', 'Yrotation', 'Xrotation', 'Zrotation']
  node = BvhNode('test_node', 'JOINT', 0, 0, np.array((0., 0., 0.)), channels)
  
  assert node.channels == channels
  
def test_init_parent():
  channels = ['Xposition', 'Yposition', 'Zposition']
  parent = BvhNode('test_p', 'JOINT', 0, 0, np.array((0., 0., 0.)), channels)
  child = BvhNode('test_c', 'JOINT', 1, 3, np.array((0., 0., 0.)), channels, parent)
  
  assert child.parent == parent
  assert child == parent.children[0]
  
def test_dof():
  channels = ['Xposition', 'Yposition', 'Zposition', 'Yrotation', 'Xrotation', 'Zrotation']
  node = BvhNode('test_node', 'JOINT', 0, 0, np.array((0., 0., 0.)), channels)
  
  assert node.dof == 6