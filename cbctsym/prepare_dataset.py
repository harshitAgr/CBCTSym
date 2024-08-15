import argparse
import os
import time

import nibabel as nib
import numpy as np
from skimage.transform import resize
from scipy.io import loadmat
import tigre
from tigre.algorithms import fdk
from tigre.utilities import CTnoise
import yaml


def load_nifti(path: str) -> np.ndarray:
    vol = nib.load(path).get_fdata().astype(np.float32)
    vol = np.swapaxes(vol, 0, 2)
    vol += 1000
    vol[vol < 0] = 0
    return vol


def write_nifti(vol: np.ndarray, path: str, im_type: str = None) -> None:
    if im_type == "proj":
        vol = vol.astype(np.float32)
    else:
        vol = vol.astype(np.int16)
        vol[vol < 0] = 0
        vol = vol.astype(np.uint16)
    vol = np.swapaxes(vol, 0, 2)
    vol = nib.Nifti1Image(vol, np.eye(4))
    nib.save(vol, path)


def threshold_vol(vol: np.ndarray, threshold: int) -> np.ndarray:
    mask = vol > threshold
    vol *= mask
    return vol


def get_geometry(vol_shape: tuple, config: dict) -> tigre.geometry:
    geo = tigre.geometry()
    geo.DSD = config["DSD"]
    geo.DSO = config["DSO"]
    geo.nDetector = np.array(config["detector_size_px"])
    geo.dDetector = np.array(config["detector_pixel_size"])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([vol_shape[0], vol_shape[1], vol_shape[2]])
    geo.sVoxel = np.array(config["image_size_mm"])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    # Offsets
    geo.offOrigin = np.array(config["image_offset_mm"])
    geo.offDetector = np.array(config["offset_detector_mm"])
    # Auxiliary
    geo.accuracy = 0.5
    # Variable to define accuracy of
    # 'interpolated' projection
    # It defines the amoutn of
    # samples per voxel.
    # Recommended <=0.5             (vx/sample)
    geo.mode = config["mode"]
    return geo


def HUtoLac(hu: np.ndarray, wLac: float) -> np.ndarray:
    return hu * wLac / 1000


def LacToHU(lac: np.ndarray, wLac: float) -> np.ndarray:
    return lac * 1000 / wLac


def material_decomposition(
    non_metal: np.ndarray, metal_vol: np.ndarray, water_threshold, bone_threshold
) -> tuple:
    metal_mask = metal_vol > 0

    water_vol, bone_vol = (
        np.zeros_like(non_metal),
        np.zeros_like(non_metal),
    )
    water_mask = non_metal < water_threshold
    bone_mask = non_metal > bone_threshold
    both_mask = ~water_mask & ~bone_mask

    water_vol[water_mask] = non_metal[water_mask]
    bone_vol[bone_mask] = non_metal[bone_mask]

    both_values = non_metal[both_mask]
    bone_vol[both_mask] = (
        both_values
        * (both_values - water_threshold)
        / (bone_threshold - water_threshold)
    )
    water_vol[both_mask] = both_values - bone_vol[both_mask]

    water_vol[metal_mask] = 0
    bone_vol[metal_mask] = 0
    return water_vol, bone_vol


def load_material_data() -> tuple:
    # load give spectrum
    spec = loadmat("material_data/GE14Spectrum120KVP.mat")["GE14Spectrum120KVP"]
    # load given LACs
    attenuation_mode = 6
    mu_water = loadmat("material_data/MiuofH2O.mat")["MiuofH2O"]
    mu_cortical_bone = loadmat("material_data/MiuofBONE_Cortical_ICRU44.mat")[
        "MiuofBONE_Cortical_ICRU44"
    ]
    mu_titanium = loadmat("material_data/MiuofTi.mat")["MiuofTi"]
    return (
        mu_water[:, attenuation_mode],
        mu_cortical_bone[:, attenuation_mode],
        mu_titanium[:, attenuation_mode],
        spec,
    )


def load_yaml(config_file: str) -> dict:
    """Loads config file if a string was passed"""
    if not isinstance(config_file, str):
        raise ValueError("The config file path must be a string.")

    try:
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {config_file} was not found.")
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing YAML file: {exc}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare metal artifact reduction dataset"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to the config file",
        default="cbctsym/config.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_yaml(args.config)
    path_to_metals = config["path_to_metal_vols"]
    path_to_data = config["path_to_non_metal_vols"]
    output_path = config["output_path"]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save the config file for future reference
    with open(os.path.join(output_path, "config.yaml"), "w") as file:
        yaml.dump(config, file)

    angles = np.linspace(0, 2 * np.pi, 360)
    effective_kv = config["effective_kv"]

    water_threshold = config["water_threshold"]
    bone_threshold = config["bone_threshold"]
    metal_threshold = config["metal_threshold"]

    time_start = time.time()

    metals = sorted(os.listdir(path_to_metals))
    data = sorted(os.listdir(path_to_data))
    n_non_metal_volumes = len(data)
    n_metal_volumes = len(metals)
    print(
        f"number of volumes without metal: {n_non_metal_volumes} and with metal: {n_metal_volumes}, we will generate a total of {n_non_metal_volumes*n_metal_volumes} volumes"
    )

    mu_water, mu_cortical_bone, mu_titanium, spec = load_material_data()
    ti_lac = mu_titanium[effective_kv] * config["metal_density"]
    w_lac = mu_water[effective_kv]

    # convert lacs to per mm
    ti_lac = ti_lac / 10
    w_lac = w_lac / 10

    water_threshold = HUtoLac(water_threshold, w_lac)
    bone_threshold = HUtoLac(bone_threshold, w_lac)

    print(
        f"water threshold: {water_threshold}, bone threshold: {bone_threshold}, metal attenuation: {ti_lac}, water lac: {w_lac}"
    )

    energies = np.array(range(19, 120))

    # later sample metals randomly
    for i in range(n_non_metal_volumes):
        for j in range(n_metal_volumes):
            non_metal_vol = load_nifti(os.path.join(path_to_data, data[i]))
            metal_vol = load_nifti(os.path.join(path_to_metals, metals[j]))
            if non_metal_vol.shape != metal_vol.shape:
                metal_vol = resize(metal_vol, non_metal_vol.shape)

            metal_vol[metal_vol < metal_threshold] = 0
            metal_vol[metal_vol > 0] = ti_lac
            # sometimes non-metal volume might has metals, so we record it too
            metal_vol[non_metal_vol > metal_threshold] = ti_lac

            non_metal_vol = HUtoLac(non_metal_vol, w_lac)

            water_vol, bone_vol = material_decomposition(
                non_metal_vol, metal_vol, water_threshold, bone_threshold
            )

            geomtery = get_geometry(non_metal_vol.shape, config)
            non_metal_length = tigre.Ax(non_metal_vol, geomtery, angles)
            water_path_length = tigre.Ax(water_vol, geomtery, angles)
            bone_path_length = tigre.Ax(bone_vol, geomtery, angles)
            metal_path_length = tigre.Ax(metal_vol, geomtery, angles)

            # calculate the attenuation
            proj_all = np.zeros_like(non_metal_length)
            for energy in energies:
                attenuation = (
                    water_path_length * mu_water[energy] / mu_water[effective_kv]
                    + bone_path_length
                    * mu_cortical_bone[energy]
                    / mu_cortical_bone[effective_kv]
                    + metal_path_length
                    * mu_titanium[energy]
                    / mu_titanium[effective_kv]
                )
                proj_all += spec[energy, 1] * np.exp(-attenuation)
            proj_all = -np.log(proj_all / spec[energies, 1].sum())

            # add noise
            proj_all = CTnoise.add(proj_all, Poisson=1e5, Gaussian=np.array([0, 10]))

            path_to_save = (
                os.path.join(output_path, data[i]).split(".")[0]
                + f"_metal_{metals[j].split('.')[0]}"
            )
            # remove extension from path to save
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            ending = f"_{data[i]}_{metals[j]}"
            if config["write_combined_projections"]:
                write_nifti(
                    proj_all,
                    os.path.join(path_to_save, f"proj_metal_corrupted_{ending}"),
                    im_type="proj",
                )
            # volumes
            metal_corrupted_vol = fdk(proj_all, geomtery, angles)
            if config["write_combined_vol"]:
                metal_corrupted_vol = LacToHU(metal_corrupted_vol, w_lac)
                write_nifti(
                    metal_corrupted_vol,
                    os.path.join(path_to_save, f"vol_metal_corrupted_{ending}"),
                )
            del proj_all, metal_corrupted_vol
            if config["write_metal_vol"]:
                non_metal_vol = LacToHU(non_metal_vol, w_lac)
                write_nifti(
                    non_metal_vol,
                    os.path.join(path_to_save, f"vol_non_metal_{ending}"),
                )
            if config["write_metal_vol"]:
                metal_vol = LacToHU(metal_vol, w_lac)
                write_nifti(
                    metal_vol, os.path.join(path_to_save, f"vol_metal_only_{ending}")
                )
            del (
                non_metal_vol,
                metal_vol,
                water_vol,
                bone_vol,
                water_path_length,
                bone_path_length,
                metal_path_length,
                non_metal_length,
            )

    print(
        f"total time taken in hours: {(time.time()-time_start)/3600} for {n_non_metal_volumes*n_metal_volumes} volumes"
    )


if __name__ == "__main__":
    main()
