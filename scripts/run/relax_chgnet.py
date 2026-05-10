"""Post-generation structure relaxation using CHGNet.

Reads a generated eval_diff file, relaxes each structure with CHGNet,
and writes a new eval_diff file with relaxed coordinates/lattices.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._bootstrap import set_default_env
set_default_env()


def tensors_to_structures(frac_coords, lattices, atom_types, num_atoms):
    batch_size = num_atoms.shape[0]
    structures = []
    offset = 0
    for i in range(batch_size):
        n = num_atoms[i].item()
        fc = frac_coords[offset:offset + n].numpy()
        lat = Lattice(lattices[i].numpy())
        types = atom_types[offset:offset + n].numpy().astype(int).tolist()
        structures.append(Structure(lat, types, fc, coords_are_cartesian=False))
        offset += n
    return structures


def structures_to_tensors(structures, original_num_atoms, device):
    all_frac_coords = []
    all_lattices = []
    for struct in structures:
        all_frac_coords.append(torch.tensor(struct.frac_coords, dtype=torch.float32))
        all_lattices.append(torch.tensor(struct.lattice.matrix, dtype=torch.float32).unsqueeze(0))

    frac_coords = torch.cat(all_frac_coords, dim=0).to(device)
    lattices = torch.cat(all_lattices, dim=0).to(device)

    from dao.common.data_utils import lattice_params_to_matrix_torch
    lengths, angles = [], []
    for struct in structures:
        lengths.append([struct.lattice.a, struct.lattice.b, struct.lattice.c])
        angles.append([struct.lattice.alpha, struct.lattice.beta, struct.lattice.gamma])
    lengths = torch.tensor(lengths, dtype=torch.float32).to(device)
    angles = torch.tensor(angles, dtype=torch.float32).to(device)

    return frac_coords, lattices, lengths, angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval_diff_*.pt file")
    parser.add_argument("--output", default="", help="Output path (default: input with _relaxed suffix)")
    parser.add_argument("--batch-size", type=int, default=1, help="Structures to relax at once")
    parser.add_argument("--max-steps", type=int, default=500, help="Max relaxation steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

    print("Loading CHGNet...")
    chgnet = CHGNet.load()
    calculator = CHGNetCalculator(model=chgnet, device=args.device)

    print(f"Loading structures from {args.input}...")
    data = torch.load(args.input, map_location="cpu", weights_only=False)

    frac_coords_all = data["frac_coords"]
    num_atoms_all = data["num_atoms"]
    atom_types_all = data["atom_types"]
    lattices_all = data["lattices"]

    num_evals = frac_coords_all.shape[0]
    total_structures = num_atoms_all.shape[1]
    print(f"Relaxing {num_evals} evals x {total_structures} structures...")

    relaxed_frac_coords = []
    relaxed_lattices = []
    relaxed_lengths = []
    relaxed_angles = []

    for eval_idx in range(num_evals):
        eval_fc, eval_lat, eval_len, eval_ang = [], [], [], []
        offset = 0

        for struct_idx in range(total_structures):
            n = num_atoms_all[eval_idx, struct_idx].item()
            fc = frac_coords_all[eval_idx, offset:offset + n]
            lat = lattices_all[eval_idx, struct_idx]
            atypes = atom_types_all[eval_idx, offset:offset + n]
            offset += n

            lattice = Lattice(lat.numpy())
            types = atypes.numpy().astype(int).tolist()
            struct = Structure(lattice, types, fc.numpy(), coords_are_cartesian=False)

            try:
                struct = struct.copy()
                struct.set_calculator(calculator)
                from ase.optimize import BFGS
                ase_struct = struct.to_ase_atoms()
                ase_struct.calc = calculator
                opt = BFGS(ase_struct, logfile=None)
                opt.run(fmax=0.05, steps=args.max_steps)
                relaxed = Structure.from_ase_atoms(ase_struct)
                eval_fc.append(torch.tensor(relaxed.frac_coords, dtype=torch.float32))
                eval_lat.append(torch.tensor(relaxed.lattice.matrix, dtype=torch.float32).unsqueeze(0))
                eval_len.append(torch.tensor([[relaxed.lattice.a, relaxed.lattice.b, relaxed.lattice.c]], dtype=torch.float32))
                eval_ang.append(torch.tensor([[relaxed.lattice.alpha, relaxed.lattice.beta, relaxed.lattice.gamma]], dtype=torch.float32))
            except Exception as e:
                print(f"  Warning: relaxation failed for eval={eval_idx} struct={struct_idx}: {e}")
                eval_fc.append(fc)
                eval_lat.append(lat.unsqueeze(0))
                from dao.common.data_utils import lattices_to_params_shape
                l, a = lattices_to_params_shape(lat.unsqueeze(0).unsqueeze(0))
                eval_len.append(l.squeeze(0))
                eval_ang.append(a.squeeze(0))

            if (struct_idx + 1) % 100 == 0:
                print(f"  eval {eval_idx}/{num_evals}: {struct_idx+1}/{total_structures} relaxed")

        relaxed_frac_coords.append(torch.cat(eval_fc, dim=0))
        relaxed_lattices.append(torch.cat(eval_lat, dim=0))
        relaxed_lengths.append(torch.cat(eval_len, dim=0))
        relaxed_angles.append(torch.cat(eval_ang, dim=0))

    data["frac_coords"] = torch.stack(relaxed_frac_coords, dim=0) if num_evals > 1 else relaxed_frac_coords[0].unsqueeze(0)
    data["lattices"] = torch.stack(relaxed_lattices, dim=0) if num_evals > 1 else relaxed_lattices[0].unsqueeze(0)
    data["lengths"] = torch.cat(relaxed_lengths, dim=0) if num_evals > 1 else relaxed_lengths[0]
    data["angles"] = torch.cat(relaxed_angles, dim=0) if num_evals > 1 else relaxed_angles[0]

    out_path = args.output or str(Path(args.input).with_name(
        Path(args.input).stem + "_relaxed" + Path(args.input).suffix))
    torch.save(data, out_path)
    print(f"Saved relaxed structures to {out_path}")


if __name__ == "__main__":
    main()
