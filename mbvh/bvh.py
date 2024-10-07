import re
import warnings
import numpy as np

class BvhNode:
  def __init__(self, name, type, id, dof_index, offset=None, channels=None, parent=None):
    self.name = name
    self.type = type
    self.id = id
    self.dof_index = dof_index
    self.offset = offset
    self.channels = channels
    self.parent = parent
    self.children = []
    if self.parent:
      self.parent.add_child(self)
        
    if self.channels:
      self.dof = len(self.channels)
    else:
      self.dof = 0
    
  def add_child(self, node):
    node.parent = self
    self.children.append(node)

class Bvh:
  def __init__(self, node_list, frame_num, s_time, frames):
    self.node_list = node_list
    self.frame_num = frame_num
    self.sampling_time = s_time
    self.frames = frames
    
    self.node_num = len(self.node_list)
    self.set_end_list()
    
  def set_end_list(self):
    self.end_list = []
    for n in self.node_list:
      if n.type == 'End':
        self.end_list.append(n)
    
  @staticmethod
  def set_node(node_list, node, id, dof_index, offset, channels):
    node_list.append(node)
    id = id + 1
    dof_index = dof_index + node.dof
    offset = []
    channels = []
    
    return node_list, id, dof_index, offset, channels, node
        
  @staticmethod
  def read_bvh(file):
    data = file
    frames = []
    text = []
    stack = ''
    for char in data:
      if char not in ('\n', '\r'):
        stack += char
      if stack:
        text.append(re.split('\\s+', stack.strip()))
        stack = ''
        
    node_list = []  
    node = None
    node_type = None
    node_name = None
    parent = None
    id = 0
    dof_index = 0
    offset = None
    channels = None
    frames = np.array([])
    
    node_init_flag = False
    node_start_flag = False
    frame_flag = [False, False]

    for item in text:
      if frame_flag[0] and frame_flag[1]:
        if not( len(frames) > 0 ):
          frames = np.empty((0, len(item)))
        frame =  np.array([item], dtype='float32')
        frames = np.append(frames, frame, axis=0)
        continue
      key = item[0]
      if key == '{':
        node_start_flag = True
        node_init_flag = False
      elif key == '}':
        if not node_init_flag and node_start_flag:
          node = BvhNode(node_name, node_type, id, dof_index, offset, channels, parent)
          node_list, id, dof_index, offset, channels, parent = Bvh.set_node(node_list, node, id, dof_index, offset, channels)
          node_init_flag = True
        if parent:
          parent = node_list[parent.id-1]
        node_start_flag = False
      elif key == 'OFFSET':
        offset = np.array(item[1:], dtype='float32')
      elif key == 'CHANNELS':
        channels = item[2:]
      elif key == 'ROOT' or key == 'JOINT' or key == 'End':
        if node_start_flag:
          node = BvhNode(node_name, node_type, id, dof_index, offset, channels, parent) 
          node_list, id, dof_index, offset, channels, parent = Bvh.set_node(node_list, node, id, dof_index, offset, channels)
          node_init_flag = True
        node_type = key
        node_name = item[1]
      elif key == 'MOTION':
        frame_flag[0] = True
      elif key == 'Frames:':
        frame_num = int(item[1])
      elif key == 'Frame' and item[1] == 'Time:':
        sampling_time = float(item[2])
        frame_flag[1] = True
        
    return Bvh(node_list, frame_num, sampling_time, frames)
  
  def get_joint(self, joint_name):
    for n in self.node_list:
      if n.name == joint_name and n.type == 'JOINT':
        return n
    warnings.warn(f"Warning: Joint '{joint_name}' is not recognized.", UserWarning)
    return None
      
  def get_joint_index(self, node_name):
    index = 0
    for n in self.node_list:
      if n.name == node_name:
        return n
      index = index + 1
      
  def get_joint_frame(self, node, frame_index):
    return self.frames[frame_index][node.dof_index:node.dof_index+node.dof]

  def calc_joint_rel_frame(self, node, frame_index):
    rel_frame = np.identity(4)
    if node.dof > 0:
      frame_vec = self.frames[frame_index][node.dof_index:node.dof_index+node.dof]
      rel_pos = np.array(node.offset)
      rel_rot = np.identity(3)

      for i in range(node.dof):
        if node.channels[i] == 'Xposition':
          rel_pos[0] = frame_vec[i]
        elif node.channels[i] == 'Yposition':
          rel_pos[1] = frame_vec[i]
        elif node.channels[i] == 'Zposition':
          rel_pos[2] = frame_vec[i]
        elif node.channels[i] == 'Xrotation':
          rot = np.zeros((3,3))
          theta = frame_vec[i]
          rot[0,0] = 1
          rot[1,1] = np.cos(theta)
          rot[1,2] = -np.sin(theta)
          rot[2,1] = np.sin(theta)
          rot[2,2] = np.cos(theta)
          rel_rot = rot @ rel_rot
        elif node.channels[i] == 'Yrotation':
          rot = np.zeros((3,3))
          theta = frame_vec[i]
          rot[0,0] = np.cos(theta)
          rot[0,2] = np.sin(theta)
          rot[1,1] = 1
          rot[2,0] = -np.sin(theta)
          rot[2,2] = np.cos(theta)
          rel_rot = rot @ rel_rot
        elif node.channels[i] == 'Zrotation':
          rot = np.zeros((3,3))
          theta = frame_vec[i]
          rot[0,0] = np.cos(theta)
          rot[0,1] = -np.sin(theta)
          rot[1,0] = np.sin(theta)
          rot[1,1] = np.cos(theta)
          rot[2,2] = 1
          rel_rot = rot @ rel_rot

      rel_frame[0:3,3] = rel_pos
      rel_frame[0:3,0:3] = rel_rot
    
    return rel_frame