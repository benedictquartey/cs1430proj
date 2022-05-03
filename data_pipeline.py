import cv2
import sys
import os
import argparse
from llff.poses.pose_utils import gen_poses  #From https://github.com/Fyusion/LLFF
 


def run_frame_extraction(vid_in, fps, out):

    outfldr = out + "/"
    if not os.path.exists(outfldr):
        os.makedirs(outfldr)

    cap= cv2.VideoCapture(vid_in)
    i=1
    file_cnt=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%int(fps) == 0:
            file_cnt +=1
            cv2.imwrite(outfldr+str(file_cnt)+'.jpg',frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()


def write_config_txt(name,scenedir):

    f = open(f"{name}_config.txt", "x")

    config = f"""expname = {name}_test\nbasedir = ./logs\ndatadir = {scenedir}\ndataset_type = llff\n\nfactor = 8\nllffhold = 8\n\nN_rand = 1024\nN_samples = 64\nN_importance = 64\n\nuse_viewdirs = True\nraw_noise_std = 1e0"""
    f.writelines(config)
    f.close()
    print(f"Done writing {name}_config.txt\n")



if __name__ == "__main__":

    modes = ["get-frames","get-poses"]

    parser = argparse.ArgumentParser(prog="pipeline", description='...')
    parser.add_argument('--expname', default='dancingQ', type=str)
    parser.add_argument('--fps', default='2', type=str)
    parser.add_argument('--video', default='sample_data/vid.mp4', type=str)
    parser.add_argument('--out', default='output', type=str)
    parser.add_argument('--match_type', type=str, default='exhaustive_matcher', help='type of matcher used.  Valid options: \
					exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')

  
    args = parser.parse_args()

    if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
        print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
        sys.exit()
    
    print("*******************************************\n Extracting Frames From Video \n*******************************************")

    #Frame extraction from video
    run_frame_extraction(args.video, args.fps, args.out+"/images")

    print("\n*******************************************\n Generating LLFF Poses From Images \n*******************************************")

    #Images to poses using COLMAP for structure from motion
    gen_poses(args.out, args.match_type)

    print("\n*******************************************\n Writing Nerf Training Config File \n*******************************************")
    #Write config txt file with parameters for training nerf
    write_config_txt(args.expname,args.out)

