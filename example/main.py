from mbvh import *

def main():
  with open('./byebye.bvh', 'r') as file:
    file_content = file.read()
  bvh = Bvh.parse_bvh(file_content)

  bvh.show_node_tree()

  bvh.create_animation()

if __name__ == "__main__":
    main()