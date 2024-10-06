import re
import numpy as np

import mathrobo as mr

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

    frame_flag = [False, False]

    node_list = []  
    node = None
    joint_type = None
    joint_name = None
    p_id = -1
    id = 0
    dof_index = 0
    offset = None
    channels = None
    frames = np.array([])
    for item in text:
      if frame_flag[0] and frame_flag[1]:
        if not( len(frames) > 0 ):
          frames = np.empty((0, len(item)))
        frame =  np.array([item], dtype='float32')
        frames = np.append(frames, frame, axis=0)
        continue
      key = item[0]
      if key == '{':
        if p_id >= 0:
          node = BvhNode(joint_name, joint_type, id, dof_index, offset, channels, node_list[p_id])
        else:
          node = BvhNode(joint_name, joint_type, id, dof_index, offset, channels)
        node_list.append(node)
        p_id = id
        id = id + 1
        dof_index = dof_index + node.dof
      elif key == '}':
        p_id = p_id - 1
      elif key == 'OFFSET':
        offset = item[1:]
      elif key == 'CHANNELS':
        channels = item[1:]
      elif key == 'ROOT' or key == 'JOINT' or key == 'End':
        joint_type = key
        joint_name = item[1]
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
      if n.name == joint_name:
        return n
      
  def get_joint_index(self, joint_name):
    index = 0
    for n in self.node_list:
      if n.name == joint_name:
        return n
      index = index + 1
      
  def get_joint_frame(self, node, frame_index):
    return self.frame[frame_index][self.dof_index:self.dof_index+len(node.channels)]

  def calc_joint_rel_frame(self, node, frame_index):
    frame_vec = self.frame[frame_index][self.dof_index:self.dof_index+len(node.channels)]

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
        rel_rot = mr.euler_x(frame_vec[i]) @ rel_rot
      elif node.channels[i] == 'Yrotation':
        rel_rot = mr.euler_y(frame_vec[i]) @ rel_rot
      elif node.channels[i] == 'Zrotation':
        rel_rot = mr.euler_z(frame_vec[i]) @ rel_rot
    
    rel_frame = np.identity(4)
    rel_frame[0:3,3] = rel_pos
    rel_frame[0:3,0:3] = rel_rot
    
    return rel_frame