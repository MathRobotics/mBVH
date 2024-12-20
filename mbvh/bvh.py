import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

            if key == "{":
                node_start_flag = True
            elif key == "}":
                if node_start_flag:
                    node = BvhNode(
                        node_name, node_type, id, dof_index, offset, channels, parent
                    )
                    node_list, id, dof_index, offset, channels, parent = (
                        Bvh.update_node_list(node_list, node, id, dof_index)
                    )
                if parent:
                    parent = parent.parent
                node_start_flag = False
            elif key == "OFFSET":
                offset = np.array(line[1:], dtype="float32")
            elif key == "CHANNELS":
                channels = line[2:]
            elif key in ["ROOT", "JOINT", "End"]:
                if node_start_flag:
                    node = BvhNode(
                        node_name, node_type, id, dof_index, offset, channels, parent
                    )
                    node_list, id, dof_index, offset, channels, parent = (
                        Bvh.update_node_list(node_list, node, id, dof_index)
                    )
                node_type, node_name = key, line[1]
            elif key == "MOTION":
                frame_flag[0] = True
            elif key == "Frames:":
                frame_num = int(line[1])
            elif key == "Frame" and line[1] == "Time:":
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
        rel_pos = np.array(node.offset)
        rel_rot = np.identity(3)
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

        rel_frame[:3, :3] = rel_rot
        rel_frame[:3, 3] = rel_pos

        return rel_frame

    @staticmethod
    def apply_rotation(current_rot, channel, angle):
        ragian = np.radians(angle)
        rot = np.eye(3)
        c, s = np.cos(ragian), np.sin(ragian)
        if channel == "Xrotation":
            rot[1:, 1:] = [[c, -s], [s, c]]
        elif channel == "Yrotation":
            rot[[0, 2], [0, 2]] = [c, c]
            rot[0, 2], rot[2, 0] = s, -s
        elif channel == "Zrotation":
            rot[:2, :2] = [[c, -s], [s, c]]
        return current_rot @ rot

    def node_kinematics(self, node, frame_index, frame_list):
        rel_frame = self.calc_relative_frame(node, frame_index)
        frame = frame_list[node.parent.id] @ rel_frame
        frame_list.append(frame)

        if node.children:
            for child in node.children:
                self.node_kinematics(child, frame_index, frame_list)

    def kinematics(self, frame_index):
        frame_list = []
        frame = self.calc_relative_frame(self.node_list[0], frame_index)
        frame_list.append(frame)
        node = self.node_list[0]

        if node.children:
            for child in node.children:
                self.node_kinematics(child, frame_index, frame_list)

        return frame_list

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

    def plot_frame(self, frame_index):
        frame_list = self.kinematics(frame_index)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # node positions
        all_positions = np.array([frame[:3, 3] for frame in frame_list])

        # drop duplicate nodes
        for node in self.node_list:
            if node.parent:
                p_id = node.parent.id
                ax.plot(
                    [frame_list[node.id][0, 3], frame_list[p_id][0, 3]],
                    [frame_list[node.id][1, 3], frame_list[p_id][1, 3]],
                    [frame_list[node.id][2, 3], frame_list[p_id][2, 3]],
                    color="black",
                )

        # get min and max bounds
        epsilon = 1e-6  # set a small value
        min_bounds = np.min(all_positions, axis=0) - epsilon
        max_bounds = np.max(all_positions, axis=0) + epsilon

        # set axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(min_bounds[0], max_bounds[0])
        ax.set_ylim(min_bounds[1], max_bounds[1])
        ax.set_zlim(min_bounds[2], max_bounds[2])

        # plot each frame
        for frame in frame_list:
            origin = frame[:3, 3]
            x_axis = frame[:3, 0]
            y_axis = frame[:3, 1]
            z_axis = frame[:3, 2]

            ax.scatter(*origin)
            ax.quiver(*origin, *x_axis, color="r", length=0.5, normalize=True)
            ax.quiver(*origin, *y_axis, color="g", length=0.5, normalize=True)
            ax.quiver(*origin, *z_axis, color="b", length=0.5, normalize=True)

        # set axes equal
        def set_axes_equal(ax):
            limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            centers = np.mean(limits, axis=1)
            max_range = np.max(np.abs(limits[:, 1] - limits[:, 0]))
            bounds = np.array([centers - max_range / 2, centers + max_range / 2]).T
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_zlim(bounds[2])

        set_axes_equal(ax)

        plt.show()

    def create_animation(self, fixed_axis=True, save_path=None):
        """
        BVH data animation for display or save.

        Args:
            fixed_axis (bool): whether to fix the axis limits. If True, the axis limits are fixed to the minimum and maximum values of all node positions. If False, the axis limits are adjusted for each frame.
            save_path (str): file path to save the animation (e.g. "animation.mp4"). None if not saving.
        """

        frame_range = range(self.frame_num)
        epsilon = 1e-6

        # plot settings
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # parent-child lines
        lines = []
        for node in self.node_list:
            if node.parent:
                (line,) = ax.plot([], [], [], color="black")
                lines.append(line)

        # node positions and axes
        scatters = []
        quivers = {"x": [], "y": [], "z": []}
        for _ in self.node_list:
            scatter = ax.scatter([], [], [])
            scatters.append(scatter)
            for axis, color in zip(["x", "y", "z"], ["r", "g", "b"]):
                quiver = ax.quiver(
                    0, 0, 0, 0, 0, 0, color=color, length=0.5, normalize=True
                )
                quivers[axis].append(quiver)

        # set axes equal
        def set_axes_equal(ax):
            limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            centers = np.mean(limits, axis=1)
            max_range = np.max(np.abs(limits[:, 1] - limits[:, 0]))
            bounds = np.array([centers - max_range / 2, centers + max_range / 2]).T
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_zlim(bounds[2])

        if fixed_axis:
            all_positions = []

            for i in frame_range:
                frame_list = self.kinematics(i)
                all_positions.extend([frame[:3, 3] for frame in frame_list])

            all_positions = np.array(all_positions)
            min_bounds = np.min(all_positions, axis=0) - epsilon
            max_bounds = np.max(all_positions, axis=0) + epsilon

            ax.set_xlim(min_bounds[0], max_bounds[0])
            ax.set_ylim(min_bounds[1], max_bounds[1])
            ax.set_zlim(min_bounds[2], max_bounds[2])

            set_axes_equal(ax)

        def update(frame_index):
            # update node positions and axes
            frame_list = self.kinematics(frame_index)

            # update parent-child lines
            for line, node in zip(lines, self.node_list):
                if node.parent:
                    p_id = node.parent.id
                    line.set_data(
                        [frame_list[node.id][0, 3], frame_list[p_id][0, 3]],
                        [frame_list[node.id][1, 3], frame_list[p_id][1, 3]],
                    )
                    line.set_3d_properties(
                        [frame_list[node.id][2, 3], frame_list[p_id][2, 3]]
                    )

            # update node positions and axes
            for i, frame in enumerate(frame_list):
                origin = frame[:3, 3]
                x_axis = frame[:3, 0]
                y_axis = frame[:3, 1]
                z_axis = frame[:3, 2]

                scatters[i]._offsets3d = (origin[0:1], origin[1:2], origin[2:3])
                quivers["x"][i].set_segments([[[*origin], [*(origin + 0.5 * x_axis)]]])
                quivers["y"][i].set_segments([[[*origin], [*(origin + 0.5 * y_axis)]]])
                quivers["z"][i].set_segments([[[*origin], [*(origin + 0.5 * z_axis)]]])

            # set axes
            if not fixed_axis:
                all_positions = np.array([frame[:3, 3] for frame in frame_list])
                min_bounds = np.min(all_positions, axis=0) - epsilon
                max_bounds = np.max(all_positions, axis=0) + epsilon
                ax.set_xlim(min_bounds[0], max_bounds[0])
                ax.set_ylim(min_bounds[1], max_bounds[1])
                ax.set_zlim(min_bounds[2], max_bounds[2])
                set_axes_equal(ax)

            return lines + scatters + quivers["x"] + quivers["y"] + quivers["z"]

        # create animation
        ani = FuncAnimation(
            fig, update, frames=frame_range, interval=self.sampling_time * 1000
        )

        # save animation
        if save_path:
            ani.save(save_path, writer="ffmpeg", fps=1/self.sampling_time)

        plt.show()
