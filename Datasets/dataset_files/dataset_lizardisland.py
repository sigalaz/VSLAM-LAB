import os
import yaml
import utm
import shutil
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from PIL import Image
from typing import Any, Final
import math

ORIGIN_UTM: Final = (332845.1897240251, 8374346.182406548)
ORIGIN_ZONE: Final = (55, 'L')
              
class LIZARDISLAND_dataset(DatasetVSLAMLab):
    """LIZARDISLAND dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "lizardisland") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.dataset_folder_raw = Path(cfg['dataset_folder_raw'])

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

        # Target image resolution
        self.target_resolution = cfg.get("target_resolution", [640, 480])

        # Sequence subsets
        self.subsets = cfg.get("subsets", {})
        self.combined = cfg.get("combined", {})

    def download_sequence_data(self, sequence_name: str) -> None:

        if sequence_name in self.subsets.keys():
            super().download_sequence_data(self.subsets.get(sequence_name)[0]) 
            self.download_subsequence(sequence_name)

        if sequence_name in self.combined.keys():
            for subset in self.combined.get(sequence_name):    
                self.download_sequence_data(subset) 
            self.download_combined_subsequence(sequence_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        if sequence_name in self.subsets.keys() or sequence_name in self.combined.keys():
            return

        sequence_path: Path = self.dataset_path / sequence_name
        rgb_folder: Path = sequence_path / 'rgb_0'
        rgb_folder_raw: Path = self.dataset_folder_raw / self._get_image_folder(sequence_name)
        csv_file: Path = self.dataset_folder_raw / self._get_csv_file(sequence_name)
        # if rgb_folder.exists():
        #      return
        
        rgb_folder.mkdir(parents=True, exist_ok=True)
        estimated_new_resolution = False
        csv_data = pd.read_csv(csv_file, header=None)
        image_filenames = csv_data[0].tolist()
        for folder in rgb_folder_raw.iterdir():
            if folder.is_dir():
                for img_path in tqdm(folder.iterdir(), desc=f"Processing images in {folder}"):
                    if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        if not img_path.name in image_filenames:
                            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                            print(img_path.name)
                            continue
                        out_path = rgb_folder / img_path.name
                        if out_path.exists():
                            continue
                        if sequence_name == "LIRS_Mar_24_GoPro1" and img_path.name == "G0013195.JPG":
                            img_path = Path("/home/alejandro/VSLAM-LAB-Benchmark/Serena_Mou_Data/LIRS_Mar_24/South_Palfrey_1/GoPro1/103GOPRO/G0013194.JPG")
                        with Image.open(img_path) as im:
                            im = im.convert("RGB")

                            if not estimated_new_resolution:
                                new_height = np.sqrt(self.target_resolution[0] * self.target_resolution[1] * im.size[1] / im.size[0])
                                new_width = self.target_resolution[0] * self.target_resolution[1] / new_height
                                new_height = int(new_height)
                                new_width = int(new_width)
                                estimated_new_resolution = True

                            im_resized = im.resize((new_width, new_height), resample=Image.BICUBIC)
                            
                            im_resized.save(out_path, quality=95, optimize=True)

    def create_rgb_csv(self, sequence_name: str) -> None:
        if sequence_name in self.subsets.keys() or sequence_name in self.combined.keys():
            return

        sequence_path: Path = self.dataset_path / sequence_name
        rgb_csv: Path = sequence_path / "rgb.csv"
        # if rgb_csv.exists():
        #     return
        
        csv_file = self.dataset_folder_raw / self._get_csv_file(sequence_name)
        csv_data = pd.read_csv(csv_file, header=None)
        image_filenames = sorted(csv_data[0].astype(str).tolist())
        image_filenames = [f"rgb_0/{name.lstrip('/')}" for name in image_filenames]
        
        step_ns = int(1e9 / 10)
        timestamps = [self._get_ts0_ns(sequence_name) + i * step_ns for i in range(len(image_filenames))]
        
        out_df = pd.DataFrame({
            "ts_rgb_0 (ns)": timestamps,
            "path_rgb_0": image_filenames,
        })

        out_df.to_csv(rgb_csv, index=False)

    def create_calibration_yaml(self, sequence_name: str) -> None:

        rgb0: dict[str, Any] = {"cam_name": "rgb_0", "cam_type": "rgb",
                "cam_model": "unknown", "focal_length": [0.0, 0.0], "principal_point": [0.0, 0.0],
                "fps": float(self.rgb_hz),
                "T_BS": np.eye(4)}
        self.write_calibration_yaml(sequence_name=sequence_name, rgb = [rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        if sequence_name in self.subsets.keys() or sequence_name in self.combined.keys():
            return

        sequence_path: Path = self.dataset_path / sequence_name
        csv_file_raw: Path = self.dataset_folder_raw / self._get_csv_file(sequence_name)
        csv_file: Path = csv_file_raw.with_suffix(".corrected.csv")
        correct_gps(csv_file_raw, csv_file)

        gt_csv: Path = sequence_path / "groundtruth.csv"

        csv_data = pd.read_csv(csv_file, header=None)
        csv_data = csv_data.sort_values(by=0, kind="mergesort").reset_index(drop=True)

        image_filenames = csv_data[0].astype(str).tolist()
        lat = csv_data[1].astype(float).tolist()
        lon = csv_data[2].astype(float).tolist()
        alt = csv_data[3].astype(float).tolist()

        zone_num, zone_letter = ORIGIN_ZONE
        tx, ty, tz = [], [], []
        for i in range(len(image_filenames)):
            easting, northing,a, b = utm.from_latlon(lat[i], lon[i],
                force_zone_number=zone_num, 
                force_zone_letter=zone_letter)
            tx.append(easting - ORIGIN_UTM[0])
            ty.append(northing - ORIGIN_UTM[1])
            tz.append(alt[i])
             
        step_ns = int(1e9 / 10)
        timestamps = [self._get_ts0_ns(sequence_name) + i * step_ns for i in range(len(image_filenames))]
        out_df = pd.DataFrame({
            "ts (ns)": timestamps,
            "tx (m)": tx,
            "ty (m)": ty,
            "tz (m)": tz,  
            "qx": [0.0]*len(timestamps),
            "qy": [0.0]*len(timestamps),
            "qz": [0.0]*len(timestamps),
            "qw": [1.0]*len(timestamps),
        })

        out_df.to_csv(gt_csv, index=False)

    def _get_image_folder(self, sequence_name: str) -> Path:
        switch = {
            "LIRS_Feb_24": "LIRS_Feb_24/South_Palfrey_1/GoPro1",
            "LIRS_Mar_24_GoPro1": "LIRS_Mar_24/South_Palfrey_1/GoPro1",
            "LIRS_Mar_24_GoPro2": "LIRS_Mar_24/South_Palfrey_1/GoPro2",
            "LIRS_Sep_25_GoPro1": "LIRS_Sep_25/GP1/images",
            "LIRS_Sep_25_GoPro2": "LIRS_Sep_25/GP2/images",
        }
        return switch[sequence_name]
    
    def _get_csv_file(self, sequence_name: str) -> Path:
        switch = {
            "LIRS_Feb_24": "LIRS_Feb_24/South_Palfrey_1/GoPro1/south_palf_1.csv",
            "LIRS_Mar_24_GoPro1": "LIRS_Mar_24/South_Palfrey_1/GoPro1/march_south_palf_1-GP1.csv",
            "LIRS_Mar_24_GoPro2": "LIRS_Mar_24/South_Palfrey_1/GoPro2/march_south_palf_1-GP2.csv",
            "LIRS_Sep_25_GoPro1": "LIRS_Sep_25/GP1/sept_south_palf_1_v3_1.csv",
            "LIRS_Sep_25_GoPro2": "LIRS_Sep_25/GP2/sept_south_palf_1_v3_2.csv",
        }
        return switch[sequence_name]
    
    def _get_ts0_ns(self, sequence_name: str) -> int:
        switch = {
            "LIRS_Feb_24": 1_000_000_000_000_000_000,
            "LIRS_Mar_24_GoPro1": 2_000_000_000_000_000_000,
            "LIRS_Mar_24_GoPro2": 3_000_000_000_000_000_000,
            "LIRS_Sep_25_GoPro1": 4_000_000_000_000_000_000,
            "LIRS_Sep_25_GoPro2": 5_000_000_000_000_000_000,
        }
        return switch[sequence_name]
    
    def download_subsequence(self, sequence_name: str) -> None:
        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        rgb_csv: Path = sequence_path / "rgb.csv"
        gt_csv: Path = sequence_path / "groundtruth.csv"
        if rgb_path.exists():
                return 
        rgb_path.mkdir(parents=True, exist_ok=True)

        parent_sequence = self.subsets.get(sequence_name)[0]
        parent_sequence_path: Path = self.dataset_path / parent_sequence
        parent_rgb_csv: Path = parent_sequence_path / "rgb.csv"
        parent_gt_csv: Path = parent_sequence_path / "groundtruth.csv"
        df_rgb = pd.read_csv(parent_rgb_csv)
        df_gt = pd.read_csv(parent_gt_csv)

        ########################
        target_image_name = self.subsets.get(sequence_name)[1]  
        matches = df_rgb.index[df_rgb["path_rgb_0"] == target_image_name].to_list()
        ref_idx = int(matches[0])
        radius = self.subsets.get(sequence_name)[2]

        n = len(df_rgb)
        left = max(0, ref_idx - int(radius))
        right = min(n - 1, ref_idx + int(radius))
        idx = np.arange(left, right + 1)

        df_rgb_sub = df_rgb.iloc[idx].copy().reset_index(drop=True)
        df_gt_sub = df_gt.iloc[idx].copy().reset_index(drop=True)
        ########################
        # target_image_name = self.subsets.get(sequence_name)[1]
        # radius = self.subsets.get(sequence_name)[2]
        # target_idx = df_rgb.index[df_rgb['path_rgb_0'] == target_image_name].tolist()
        # ref_idx = target_idx[0]

        # ref_x = df_gt.at[ref_idx, 'tx (m)']
        # ref_y = df_gt.at[ref_idx, 'ty (m)']
        # ref_z = df_gt.at[ref_idx, 'tz (m)']
        # distances = np.sqrt(
        #     (df_gt['tx (m)'] - ref_x)**2 + 
        #     (df_gt['ty (m)'] - ref_y)**2 + 
        #     (df_gt['tz (m)'] - ref_z)**2
        # )
        # mask = distances <= radius

        # df_rgb_sub = df_rgb[mask].copy().reset_index(drop=True)
        # df_gt_sub = df_gt[mask].copy().reset_index(drop=True)
        ########################

        for _, row in df_rgb_sub.iterrows():
            rel_path = row['path_rgb_0']
            full_src = os.path.abspath(parent_sequence_path / rel_path)
            full_dst = os.path.abspath(sequence_path / rel_path)
            if os.path.exists(full_dst) or os.path.islink(full_dst):
                os.remove(full_dst)      
            os.symlink(full_src, full_dst)
        df_rgb_sub.to_csv(rgb_csv, index=False, sep=',') 
        df_gt_sub.to_csv(gt_csv, index=False, sep=',')

    def download_combined_subsequence(self, sequence_name):
        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        rgb_csv: Path = sequence_path / "rgb.csv"
        gt_csv: Path = sequence_path / "groundtruth.csv"
        if rgb_path.exists():
                return 
        rgb_path.mkdir(parents=True, exist_ok=True)

        dfs_rgb = []
        dfs_pose = []
        for subset in self.combined.get(sequence_name):   
            if f"s{sequence_name[-2:]}" not in subset:
                continue; 
            parent_sequence_path = self.dataset_path / subset
            parent_rgb_csv = parent_sequence_path / "rgb.csv"
            parent_gt_csv = parent_sequence_path / "groundtruth.csv"
            dfs_rgb.append(pd.read_csv(parent_rgb_csv))
            dfs_pose.append(pd.read_csv(parent_gt_csv))
            for _, row in dfs_rgb[-1].iterrows():
                rel_path = row['path_rgb_0']
                full_src = os.path.abspath(os.path.join(parent_sequence_path, rel_path))
                full_dst = os.path.abspath(os.path.join(sequence_path, rel_path))
                if os.path.exists(full_dst) or os.path.islink(full_dst):
                    os.remove(full_dst)      
                os.symlink(full_src, full_dst)
            df_rgb_all = pd.concat(dfs_rgb, ignore_index=True)
            df_pose_all = pd.concat(dfs_pose, ignore_index=True)
            df_rgb_all.to_csv(rgb_csv, index=False, sep=',') 
            df_pose_all.to_csv(gt_csv, index=False, sep=',')

def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0  # meters
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def correct_gps(in_csv, out_csv) -> None:
    T = pd.read_csv(in_csv, header=None)
    T = T.sort_values(T.columns[0]).reset_index(drop=True)
    n = len(T)
    lat_col = 1
    lon_col = 2
    
    errors = np.full(n - 1, np.nan, dtype=float)
    for i in range(n - 1):
        lat1 = float(T.iat[i, lat_col])
        lon1 = float(T.iat[i, lon_col])
        lat2 = float(T.iat[i + 1, lat_col])
        lon2 = float(T.iat[i + 1, lon_col])
        d_m = haversine_distance_m(lat1, lon1, lat2, lon2)
        if d_m > 0.6:
            d_m = -1.0
        errors[i] = d_m

    errors2 = np.full(n - 1, np.nan, dtype=float)
    for i in range(n - 1, 0, -1):
        lat1 = float(T.iat[i, lat_col])
        lon1 = float(T.iat[i, lon_col])
        lat2 = float(T.iat[i - 1, lat_col])
        lon2 = float(T.iat[i - 1, lon_col])

        d_m = haversine_distance_m(lat1, lon1, lat2, lon2)
        if d_m > 0.6:
            d_m = -1.0
        errors2[i - 1] = d_m

    both_bad = (errors[1:] == -1.0) & (errors2[:-1] == -1.0)
    error_1 = errors[1:].copy()
    error_1[both_bad] = -1.0
    errors = np.concatenate(([errors[0]], error_1))

    idx_bad = np.where(errors == -1.0)[0]  # 0-based indices into errors

    prev_idx = np.full(idx_bad.shape, np.nan)
    prev_val = np.full(idx_bad.shape, np.nan)
    next_idx = np.full(idx_bad.shape, np.nan)
    next_val = np.full(idx_bad.shape, np.nan)

    for k, i in enumerate(idx_bad):
        # previous non -1
        prev_candidates = np.where(errors[:i] != -1.0)[0]
        if prev_candidates.size > 0:
            j = prev_candidates[-1]
            prev_idx[k] = j
            prev_val[k] = errors[j]

        # next non -1
        next_candidates = np.where(errors[i + 1:] != -1.0)[0]
        if next_candidates.size > 0:
            j = next_candidates[0] + (i + 1)
            next_idx[k] = j
            next_val[k] = errors[j]

    # --- Mark rows in T involved in any bad segment
    bad_seg = (errors == -1.0)              # length n-1
    bad_row = np.zeros(n, dtype=bool)
    bad_row[:-1] |= bad_seg                 # row i touched by segment i
    bad_row[1:]  |= bad_seg                 # row i+1 touched by segment i

    # --- Set bad rows lat/lon to NaN
    lat = T.iloc[:, lat_col].astype(float).to_numpy()
    lon = T.iloc[:, lon_col].astype(float).to_numpy()
    lat[bad_row] = np.nan
    lon[bad_row] = np.nan

    # --- Interpolate using RELATIVE INDEX (row number)
    x = np.arange(n, dtype=float)

    def interp_with_nearest_edges(y: np.ndarray) -> np.ndarray:
        good = ~np.isnan(y)
        if good.sum() == 0:
            return y  # nothing to do
        y_out = y.copy()
        y_out[~good] = np.interp(x[~good], x[good], y[good])  # linear; edges become nearest by default
        return y_out

    lat = interp_with_nearest_edges(lat)
    lon = interp_with_nearest_edges(lon)

    # Put back into T
    T.iloc[:, lat_col] = lat
    T.iloc[:, lon_col] = lon

    # Save
    T.to_csv(out_csv, index=False, header=False)


