import re
import warnings
import numpy as np


class BvhNode:
    def __init__(
        self, name, type, id, dof_index, offset=None, channels=None, parent=None
    ):
        self.name = name
        self.type = type
        self.id = id
        self.dof_index = dof_index
        self.offset = offset
        self.channels = channels
        self.parent = parent
        self.children = []
        if self.parent:
            self.parent.children.append(self)

        if self.channels:
            self.dof = len(self.channels)
        else:
            self.dof = 0

class Bvh:
    def __init__(self, node_list, frame_num, sampling_time, frames):
        self.node_list = node_list
        self.frame_num = frame_num
        self.sampling_time = sampling_time
        self.frames = frames
        self.node_num = len(node_list)
        self.end_list = [n for n in node_list if n.type == "End"]

    @staticmethod
    def update_node_list(node_list, node, id, dof_index):
        node_list.append(node)
        return node_list, id + 1, dof_index + node.dof, None, None, node

    @staticmethod
    def parse_bvh(file_content):
      text_lines = [
            re.split(r"\s+", line.strip())
            for line in file_content.splitlines()
            if line.strip()
        ]
          
      node_list, frames = [], None
      parent, offset, channels = None, None, None
      node, node_type, node_name = None, None, None
      id, dof_index = 0, 0
      
      node_start_flag = False
      frame_flag = [False, False]

      # Parse each line
      for line in text_lines:
        key = line[0]

        if frame_flag[0] and frame_flag[1]:
            frame = np.array([line], dtype="float32")
            if frames is None:
                frames = frame
            else:
                frames = np.append(frames, frame, axis=0)
            continue

        if key == '{':
          node_start_flag = True
        elif key == '}':
          if node_start_flag:
            node = BvhNode(
                node_name, node_type, id, dof_index, offset, channels, parent
            )
            node_list, id, dof_index, offset, channels, parent = Bvh.update_node_list(
                node_list, node, id, dof_index
            )
          if(parent):
            parent = parent.parent
          node_start_flag = False
        elif key == 'OFFSET':
          offset = np.array(line[1:], dtype='float32')
        elif key == 'CHANNELS':
          channels = line[2:]
        elif key in ["ROOT", "JOINT", "End"]:
          if node_start_flag:
            node = BvhNode(
                node_name, node_type, id, dof_index, offset, channels, parent
            )
            node_list, id, dof_index, offset, channels, parent = Bvh.update_node_list(
                node_list, node, id, dof_index
            )
          node_type, node_name = key, line[1]
        elif key == 'MOTION':
          frame_flag[0] = True
        elif key == 'Frames:':
          frame_num = int(line[1])
        elif key == 'Frame' and line[1] == 'Time:':
          sampling_time = float(line[2])
          frame_flag[1] = True
          
      return Bvh(node_list, frame_num, sampling_time, frames)

    def get_joint(self, joint_name):
        for n in self.node_list:
            if n.name == joint_name and n.type == "JOINT":
                return n
        warnings.warn(f"Warning: Joint '{joint_name}' is not recognized.", UserWarning)
        return None

    def get_node_id(self, node_name):
        for n in self.node_list:
            if n.name == node_name:
                return n.id
        warnings.warn(f"Warning: Node '{node_name}' is not recognized.", UserWarning)
        return 0

    def get_node_dof_index(self, node_name):
        for n in self.node_list:
            if n.name == node_name:
                return n.dof_index
        warnings.warn(f"Warning: Node '{node_name}' is not recognized.", UserWarning)
        return 0

    def get_node_frame(self, node, frame_index):
        return self.frames[frame_index][node.dof_index : node.dof_index + node.dof]

    def calc_relative_frame(self, node, frame_index):
        rel_frame = np.identity(4)
        frame_values = self.frames[
            frame_index, node.dof_index : node.dof_index + node.dof
        ]
        offset_pos, offset_rot = np.array(node.offset), np.identity(3)
        rel_pos, rel_rot = np.zeros(3), np.identity(3)
        if node.dof > 0:
            for i, channel in enumerate(node.channels):
                value = frame_values[i]
                if channel == "Xposition":
                    rel_pos[0] = value
                elif channel == "Yposition":
                    rel_pos[1] = value
                elif channel == "Zposition":
                    rel_pos[2] = value
                elif channel in ["Xrotation", "Yrotation", "Zrotation"]:
                    rel_rot = self.apply_rotation(rel_rot, channel, value)

        rel_frame[:3, :3] = offset_rot @ rel_rot
        rel_frame[:3, 3] = offset_pos + offset_rot @ rel_pos

        return rel_frame

    @staticmethod
    def apply_rotation(current_rot, channel, angle):
        rot = np.eye(3)
        c, s = np.cos(angle), np.sin(angle)
        if channel == "Xrotation":
            rot[1:, 1:] = [[c, -s], [s, c]]
        elif channel == "Yrotation":
            rot[[0, 2], [0, 2]] = [c, c]
            rot[0, 2], rot[2, 0] = s, -s
        elif channel == "Zrotation":
            rot[:2, :2] = [[c, -s], [s, c]]
        return rot @ current_rot
    
    def show_node_tree(self):
        for node in self.node_list:
            print(f"{node.id}: {node.name} ({node.type})")
            if node.parent:
                print(f"  Parent: {node.parent.name}")
            if node.children:
                print("  Children:")
                for child in node.children:
                    print(f"    {child.name}")
            if node.offset is not None:
                print(f"  Offset: {node.offset}")
            if node.channels:
                print(f"  Channels: {node.channels}")
            print()

    def show_frame(self, frame_index):
        for node in self.node_list:
            print(f"{node.name}: {self.get_node_frame(node, frame_index)}")
        print()
