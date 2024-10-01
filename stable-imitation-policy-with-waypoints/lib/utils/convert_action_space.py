import argparse

SIM_BASE_POS = [-0.66, 0, 0.912]


def main(args):
    with open(args.input, 'r') as f:
        lines = f.readlines()

    # format of lines :action_num, segment_num, x_pos, y_pos, z_pos, x_euler, y_euler, z_euler
    # replace x_pos, y_pos, z_pos with [x_pos, y_pos, z_pos]-SIM_BASE_POS
    new_lines = []
    first_line = True
    for line in lines:
        if first_line:
            new_lines.append(line)
            first_line = False
            continue
        line = line.strip().split(',')
        new_line = line[:2] + [f"{float(line[2]) - SIM_BASE_POS[0] - 0.20 :.6f}", f"{float(line[3]) - SIM_BASE_POS[1]:.6f}", f"{float(line[4]) - SIM_BASE_POS[2]:.6f}"] + line[5:]
        new_lines.append(','.join(new_line) + '\n')
    
    with open(args.output, 'w') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", 
                        type=str, 
                        default=None,
                        help="Path to the input CSV file")
    parser.add_argument("-o", "--output",
                        type=str,
                        default=None,
                        help="Path to the output CSV file")
    args = parser.parse_args()                        
    main(args)  