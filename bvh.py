import re
import numpy as np

class BvhNode:
  def __init__(self, name, type, id, offset=None, channels=None, parent=None):
    self.name = name
    self.type = type
    self.id = id
    self.offset = offset
    self.channels = channels
    self.parent = parent
    self.children = []
    if self.parent:
        self.parent.add_child(self)
    
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
    offset = None
    channel = None
    frames = np.array([])
    p_id = -1
    id = 0
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
          node = BvhNode(joint_name, joint_type, id, offset, channel, node_list[p_id])
        else:
          node = BvhNode(joint_name, joint_type, id, offset, channel)
        node_list.append(node)
        p_id = id
        id = id + 1
      elif key == '}':
        p_id = p_id - 1
      elif key == 'OFFSET':
        offset = item[1:]
      elif key == 'CHANNELS':
        channel = item[1:]
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