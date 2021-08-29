#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
from os.path import join as pjoin
import shutil

from depth_fine_tuning import DepthFineTuner
from flow import Flow
from scale_calibration import calibrate_scale, check_frames
from consistent_depth.tools import make_video as mkvid
from utils.frame_range import FrameRange, OptionalSet
from utils.helpers import print_banner, print_title
# from video import (Video, sample_pairs)

import sys
from utils.helpers import mkdir_ifnotexists
from utils import (frame_sampling, image_io)
import cv2
import numpy as np

import glob
import re

ffmpeg = "ffmpeg"
ffprobe = "ffprobe"




class DatasetProcessor:
    def __init__(self, writer=None):
        self.writer = writer
        self.frame_count=0
        self.frame_path=None

    def create_output_path(self, params):
        range_tag = f"R{params.frame_range.name}"
        flow_ops_tag = "-".join(params.flow_ops)
        name = f"{range_tag}_{flow_ops_tag}_{params.model_type}"

        out_dir = pjoin(self.path, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def downscale_frames(
        self, subdir, max_size, ext, align=16, full_subdir="color_full"
    ):
        # full_dir = pjoin(self.path, full_subdir)
        full_dir = self.frame_path
        down_dir = pjoin(self.path, subdir)

        mkdir_ifnotexists(down_dir)

        if check_frames(down_dir, ext):
            # Frames are already extracted and checked OK.
            return

        frame_fls = glob.glob(os.path.join(full_dir, '*_human.png'))
        frame_fls.sort(key=lambda f: int(re.sub('\D', '', f)))

        for i, frame_fl in enumerate(frame_fls):
            name = os.path.splitext(os.path.basename(frame_fl))[0][:-6]
            full_file = os.path.join(full_dir,name + '.png')
            down_file = os.path.join(down_dir,name + '.'+ ext)

        # for i in range(self.frame_count):
        #     full_file = "%s/frame_%06d.png" % (full_dir, i)
        #     down_file = ("%s/frame_%06d." + ext) % (down_dir, i)
            suppress_messages = (i > 0)
            image = image_io.load_image(
                full_file, max_size=max_size, align=align,
                suppress_messages=suppress_messages
            )
            image = image[..., ::-1]  # Channel swizzle

            if ext == "raw":
                image_io.save_raw_float32_image(down_file, image)
            else:
                cv2.imwrite(down_file, image * 255)

        check_frames(down_dir, ext)

    def sample_pairs(self, frame_range, flow_ops):
        # TODO: update the frame range with reconstruction range
        name_mode_map = frame_sampling.SamplePairsMode.name_mode_map()
        opts = [
            frame_sampling.SamplePairsOptions(mode=name_mode_map[op]) for op in flow_ops
        ]
        pairs = frame_sampling.SamplePairs.sample(
            opts, frame_range=frame_range, two_way=True
        )
        print(f"Sampled {len(pairs)} frame pairs.")
        return pairs

    def pipeline(self, params):
        # self.extract_frames(params)
        print_banner("Downscaling frames (raw)")
        self.downscale_frames("color_down", params.size, "raw")

        print_banner("Downscaling frames (png)")
        self.downscale_frames("color_down_png", params.size, "png")

        print_banner("Downscaling frames (for flow)")
        self.downscale_frames("color_flow", Flow.max_size(), "png", align=64)

        frame_range = FrameRange(
            frame_range=params.frame_range.set, num_frames=self.frame_count,start_frame=int(os.path.basename(os.path.dirname(params.path)).split('_')[0])-1
        )
        frames = frame_range.frames()

        print_banner("Compute initial depth")

        ft = DepthFineTuner(self.out_dir, frames, params)
        initial_depth_dir = pjoin(self.path, f"depth_{params.model_type}")
        if not check_frames(pjoin(initial_depth_dir, "depth"), "raw"):
            ft.save_depth(initial_depth_dir)

        valid_frames = calibrate_scale(self.path,self.out_dir, frame_range, self.frame_path, self.frame_count, params)
        # frame range for finetuning:
        ft_frame_range = frame_range.intersection(OptionalSet(set(valid_frames)))
        print("Filtered out frames",
            sorted(set(frame_range.frames()) - set(ft_frame_range.frames())))

        print_banner("Compute flow")

        frame_pairs = self.sample_pairs(ft_frame_range, params.flow_ops)
        self.flow.compute_flow(frame_pairs, params.flow_checkpoint)

        print_banner("Compute flow masks")

        self.flow.mask_valid_correspondences()

        flow_list_path = self.flow.check_good_flow_pairs(
            frame_pairs, params.overlap_ratio
        )
        shutil.copyfile(flow_list_path, pjoin(self.path, "flow_list.json"))

        print_banner("Visualize flow")

        self.flow.visualize_flow(warp=True)

        print_banner("Fine-tuning")

        if not check_frames(pjoin(ft.out_dir, "depth"), "raw", frames):
            ft.fine_tune(writer=self.writer,valid_frames=list(valid_frames))
            print_banner("Compute final depth")
            ft.save_depth(ft.out_dir, frames)

        if params.make_video and (not os.path.isdir(pjoin(os.path.dirname(ft.out_dir), "videos"))):
            print_banner("Export visualization videos")
            self.make_videos(params, ft.out_dir)

        return initial_depth_dir, ft.out_dir, frame_range.frames()

    def process(self, params):
        self.path = params.path
        self.frame_path=params.frame_path
        self.frame_count=params.frame_count
        os.makedirs(self.path, exist_ok=True)

        # self.video_file = params.video_file

        self.out_dir = self.create_output_path(params)

        # self.video = Video(params.path, params.video_file)
        self.flow = Flow(params.path, self.out_dir)

        print_title(f"Processing dataset '{self.path}'")

        print(f"Output directory: {self.out_dir}")

        if not os.path.exists(os.path.join(self.out_dir,'metadata_scaled.npz')):
            if params.op == "all":
                return self.pipeline(params)
            # elif params.op == "extract_frames":
            #     return self.extract_frames(params)
            else:
                raise RuntimeError("Invalid operation specified.")
        else:
            print("Output directory exists (Skipped)")

    def make_videos(self, params, ft_depth_dir):
        args = [
            "--color_dir", pjoin(self.path, "color_down_png"),
            "--out_dir", pjoin(self.out_dir, "videos"),
            "--depth_dirs",
            pjoin(self.path, f"depth_{params.model_type}"),
            pjoin(self.path, "depth_colmap_dense"),
            pjoin(ft_depth_dir, "depth"),
        ]
        gt_dir = pjoin(self.path, "depth_gt")
        if os.path.isdir(gt_dir):
            args.append(gt_dir)

        vid_params = mkvid.MakeVideoParams().parser.parse_args(
            args,
            namespace=params
        )
        logging.info("Make videos {}".format(vid_params))
        mkvid.main(vid_params)
