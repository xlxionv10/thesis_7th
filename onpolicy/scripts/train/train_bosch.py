#!/usr/bin/env python
import sys
import os
import multiprocessing as mp
import platform
import json
import socket
import numpy as np
from pathlib import Path
import torch

# Allow running this file directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import wandb
except ImportError:
    wandb = None

try:
    import setproctitle
except ImportError:
    setproctitle = None
try:
    import yaml
except ImportError:
    yaml = None

from onpolicy.config import get_config
from onpolicy.envs.bosch import BoschEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def _load_bosch_config(config_path):
    if config_path is None:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Bosch config file not found: {config_path}")

    suffix = Path(config_path).suffix.lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to load YAML configs. Install with `pip install pyyaml`."
                )
            data = yaml.safe_load(f)
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(
                "Unsupported config format. Use .json or .yaml/.yml."
            )

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Bosch config must be a JSON/YAML object (dict).")
    return data


def _save_bosch_config(config_path, cfg):
    suffix = Path(config_path).suffix.lower()
    with open(config_path, "w", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to save YAML configs. Install with `pip install pyyaml`."
                )
            yaml.safe_dump(cfg, f, sort_keys=False)
        elif suffix == ".json":
            json.dump(cfg, f, indent=2)
        else:
            raise ValueError(
                "Unsupported config format. Use .json or .yaml/.yml."
            )


def _auto_fill_bosch_config(cfg, seed=1):
    rng = np.random.RandomState(int(seed))

    num_lines = int(cfg.get("num_lines", 6))
    num_products = int(cfg.get("num_products", 3))
    num_periods = int(cfg.get("num_periods", 12))
    scale = max(0.5, float(num_lines) / 6.0)

    def maybe_set(key, value):
        if cfg.get(key) is None:
            cfg[key] = value
            return True
        return False

    filled = []

    # Capacity per line
    if maybe_set("capacity_per_line", (np.ones(num_lines) * 23.83).tolist()):
        filled.append("capacity_per_line")

    # Eligibility matrix (ensure each line/product has at least one)
    if cfg.get("eligibility_matrix") is None:
        elig = (rng.rand(num_lines, num_products) < 0.7).astype(int)
        for l in range(num_lines):
            if elig[l].sum() == 0:
                elig[l, rng.randint(0, num_products)] = 1
        for p in range(num_products):
            if elig[:, p].sum() == 0:
                elig[rng.randint(0, num_lines), p] = 1
        cfg["eligibility_matrix"] = elig.tolist()
        filled.append("eligibility_matrix")

    # Processing time matrix (hours per unit)
    if cfg.get("processing_time_matrix") is None:
        base = rng.uniform(0.0055, 0.0078, size=num_products)
        line_factor = rng.uniform(0.95, 1.10, size=num_lines)
        mat = np.outer(line_factor, base)
        mat += rng.normal(0.0, 0.0002, size=mat.shape)
        mat = np.clip(mat, 0.0050, 0.0090)
        cfg["processing_time_matrix"] = mat.round(6).tolist()
        filled.append("processing_time_matrix")

    # Production cost matrix
    if cfg.get("production_cost_matrix") is None:
        base = rng.uniform(0.9, 1.4, size=num_products)
        line_factor = rng.uniform(0.95, 1.15, size=num_lines)
        mat = np.outer(line_factor, base)
        mat += rng.normal(0.0, 0.05, size=mat.shape)
        mat = np.clip(mat, 0.6, 2.5)
        cfg["production_cost_matrix"] = mat.round(3).tolist()
        filled.append("production_cost_matrix")

    # Setup time matrix (lines x products x products)
    if cfg.get("setup_time_matrix") is None:
        time_range = cfg.get("setup_time_offdiag_range", [0.3, 1.4])
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            time_range = [0.3, 1.4]
        t_low, t_high = float(time_range[0]), float(time_range[1])
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        diag_val = float(cfg.get("setup_time_diag", 0.0))
        base = rng.uniform(t_low, t_high, size=(num_products, num_products))
        np.fill_diagonal(base, diag_val)
        line_factor = rng.uniform(0.9, 1.2, size=num_lines)
        mat = np.stack([base * f for f in line_factor], axis=0)
        mat = np.clip(mat + rng.normal(0.0, 0.05, size=mat.shape), t_low, t_high)
        for l in range(num_lines):
            np.fill_diagonal(mat[l], diag_val)
        cfg["setup_time_matrix"] = mat.round(3).tolist()
        filled.append("setup_time_matrix")

    # Setup cost matrix (tracks setup time by default)
    if cfg.get("setup_cost_matrix") is None:
        cost_per_hour = float(cfg.get("setup_cost_per_hour", 1.0))
        if cfg.get("setup_time_matrix") is not None:
            mat = np.array(cfg["setup_time_matrix"], dtype=np.float32) * cost_per_hour
            cfg["setup_cost_matrix"] = mat.round(3).tolist()
        else:
            base = rng.uniform(2.0, 7.0, size=(num_products, num_products))
            np.fill_diagonal(base, 0.0)
            line_factor = rng.uniform(0.9, 1.2, size=num_lines)
            mat = np.stack([base * f for f in line_factor], axis=0)
            mat = np.clip(mat + rng.normal(0.0, 0.5, size=mat.shape), 0.0, 12.0)
            cfg["setup_cost_matrix"] = mat.round(2).tolist()
        filled.append("setup_cost_matrix")

    # First setup cost/time (per line)
    if maybe_set(
        "first_setup_cost",
        rng.uniform(4.0, 8.0, size=num_lines).round(2).tolist(),
    ):
        filled.append("first_setup_cost")
    if maybe_set(
        "first_setup_time",
        rng.uniform(0.8, 1.6, size=num_lines).round(2).tolist(),
    ):
        filled.append("first_setup_time")

    # Hazard rates and maintenance costs/times
    if maybe_set(
        "hazard_rate",
        rng.uniform(0.0008, 0.0025, size=num_lines).round(6).tolist(),
    ):
        filled.append("hazard_rate")
    if maybe_set(
        "pm_cost",
        rng.uniform(16.0, 28.0, size=num_lines).round(2).tolist(),
    ):
        filled.append("pm_cost")
    if maybe_set(
        "cm_cost",
        rng.uniform(30.0, 50.0, size=num_lines).round(2).tolist(),
    ):
        filled.append("cm_cost")
    if maybe_set(
        "pm_time",
        rng.uniform(1.0, 2.2, size=num_lines).round(2).tolist(),
    ):
        filled.append("pm_time")
    if maybe_set(
        "cm_time",
        rng.uniform(7.0, 12.0, size=num_lines).round(2).tolist(),
    ):
        filled.append("cm_time")

    # Demand profile (periods x products)
    if cfg.get("demand_profile") is None:
        t = np.arange(num_periods, dtype=np.float32)
        profile = np.zeros((num_periods, num_products), dtype=np.float32)

        # Compute per-product capacity caps when possible.
        cap_units = None
        try:
            elig = np.array(cfg.get("eligibility_matrix"), dtype=np.float32)
            proc = np.array(cfg.get("processing_time_matrix"), dtype=np.float32)
            cap = np.array(cfg.get("capacity_per_line"), dtype=np.float32)
            if (
                elig.shape == proc.shape
                and elig.shape[0] == cap.shape[0]
                and elig.shape[1] == num_products
            ):
                cap_units = np.zeros(num_products, dtype=np.float32)
                for p in range(num_products):
                    mask = (elig[:, p] > 0.5) & (proc[:, p] > 0.0)
                    if np.any(mask):
                        cap_units[p] = float(
                            np.sum(cap[mask] / proc[mask, p])
                        )
        except Exception:
            cap_units = None

        demand_range = cfg.get("demand_range")
        if isinstance(demand_range, (list, tuple)) and len(demand_range) == 2:
            dmin, dmax = float(demand_range[0]), float(demand_range[1])
            if dmin > dmax:
                dmin, dmax = dmax, dmin
            for p in range(num_products):
                dmax_p = dmax
                if cap_units is not None and cap_units[p] > 0.0:
                    dmax_p = min(dmax_p, float(cap_units[p]))
                dmin_p = min(dmin, dmax_p)
                # Build a seasonal range bounded by [dmin_p, dmax_p].
                low_min = dmin_p
                low_max = dmin_p + 0.3 * (dmax_p - dmin_p)
                high_min = dmin_p + 0.7 * (dmax_p - dmin_p)
                high_max = dmax_p
                low = rng.uniform(low_min, low_max) if low_max >= low_min else low_min
                high = rng.uniform(high_min, high_max) if high_max >= high_min else high_max
                phase = rng.uniform(0, 2 * np.pi)
                seasonal = 0.5 * (
                    1.0
                    + np.sin(
                        2 * np.pi * t / max(1, num_periods) + phase
                    )
                )
                demand = low + (high - low) * seasonal
                profile[:, p] = np.round(
                    np.clip(demand, dmin_p, dmax_p)
                )
        else:
            for p in range(num_products):
                low = rng.uniform(800, 1500) * scale
                high = low + rng.uniform(800, 2200) * scale
                if cap_units is not None and cap_units[p] > 0.0:
                    high = min(high, float(cap_units[p]))
                phase = rng.uniform(0, 2 * np.pi)
                seasonal = 0.5 * (
                    1.0
                    + np.sin(
                        2 * np.pi * t / max(1, num_periods) + phase
                    )
                )
                demand = low + (high - low) * seasonal
                profile[:, p] = np.round(demand)
        cfg["demand_profile"] = profile.astype(int).tolist()
        filled.append("demand_profile")

    return cfg, filled


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BOSCH":
                env = BoschEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        )


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BOSCH":
                env = BoschEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    """
    Add Bosch-specific arguments on top of the common config.
    """
    parser.add_argument(
        "--bosch_config",
        type=str,
        default=None,
        help="Path to BOSCH scenario config (.json/.yaml).",
    )
    parser.add_argument("--num_lines", type=int, default=6)
    parser.add_argument("--num_products", type=int, default=3)
    parser.add_argument("--num_periods", type=int, default=24)
    parser.add_argument(
        "--capacity_per_line",
        type=str,
        default="100.0",
        help="Per-line capacity (comma-separated) or scalar.",
    )
    parser.add_argument("--max_lot_size", type=int, default=10)
    parser.add_argument(
        "--manager_max_horizon",
        type=int,
        default=7,
        help="Max number of days of demand to cover per line (manager action).",
    )

    parser.add_argument("--holding_cost", type=float, default=1.0)
    parser.add_argument("--backlog_cost", type=float, default=10.0)
    parser.add_argument("--production_cost", type=float, default=1.0)
    parser.add_argument("--setup_cost", type=float, default=2.0)
    parser.add_argument(
        "--pm_cost",
        type=str,
        default="20.0",
        help="Per-line PM cost (comma-separated) or scalar.",
    )
    parser.add_argument(
        "--cm_cost",
        type=str,
        default="40.0",
        help="Per-line CM cost (comma-separated) or scalar.",
    )
    parser.add_argument("--alpha_cost_weight", type=float, default=0.1)
    parser.add_argument(
        "--hazard_rate",
        type=str,
        default="1e-3",
        help="Per-line hazard rates (comma-separated) or scalar.",
    )

    # Time-based parameters
    parser.add_argument(
        "--pm_time",
        type=str,
        default="0.0",
        help="Per-line PM time (comma-separated) or scalar.",
    )
    parser.add_argument(
        "--cm_time",
        type=str,
        default="0.0",
        help="Per-line CM time (comma-separated) or scalar.",
    )

    # Comma-separated lists for per-product parameters
    parser.add_argument(
        "--processing_time",
        type=str,
        default="1.0",
        help="Per-product processing time (hours per unit), comma-separated or scalar.",
    )
    parser.add_argument(
        "--processing_time_matrix",
        type=str,
        default=None,
        help="Flattened line×product processing time matrix (row-major).",
    )
    parser.add_argument(
        "--mean_demand",
        type=str,
        default="10.0",
        help="Per-product mean demand per period, comma-separated or scalar.",
    )

    # Optional scalar or matrices for setup and production
    parser.add_argument(
        "--setup_time",
        type=float,
        default=0.0,
        help="Base setup time (hours) for switching between different products.",
    )
    parser.add_argument(
        "--setup_cost_matrix",
        type=str,
        default=None,
        help="Flattened product×product setup cost matrix (row-major).",
    )
    parser.add_argument(
        "--setup_time_matrix",
        type=str,
        default=None,
        help="Flattened product×product setup time matrix (row-major).",
    )
    parser.add_argument(
        "--first_setup_cost",
        type=str,
        default=None,
        help="Initial setup cost when a line with no prior product starts production.",
    )
    parser.add_argument(
        "--first_setup_time",
        type=str,
        default=None,
        help="Initial setup time when a line with no prior product starts production.",
    )
    parser.add_argument(
        "--production_cost_matrix",
        type=str,
        default=None,
        help="Flattened line×product production cost matrix (row-major).",
    )
    parser.add_argument(
        "--eligibility_matrix",
        type=str,
        default=None,
        help="Flattened line×product eligibility matrix (0/1, row-major).",
    )
    parser.add_argument(
        "--demand_profile",
        type=str,
        default=None,
        help="Flattened period×product demand profile (row-major).",
    )
    parser.add_argument(
        "--max_actions_per_period",
        type=int,
        default=8,
        help="Maximum number of machine micro-actions per period.",
    )
    parser.add_argument(
        "--dense_production_reward",
        type=float,
        default=1.0,
        help="Immediate reward weight per produced unit for machine agents.",
    )
    parser.add_argument(
        "--dense_setup_penalty",
        type=float,
        default=1.0,
        help="Immediate penalty weight for setup cost during changeovers.",
    )
    parser.add_argument(
        "--dense_pm_penalty",
        type=float,
        default=1.0,
        help="Immediate penalty weight for PM cost when PM action is taken.",
    )
    parser.add_argument(
        "--debug_actions",
        action="store_true",
        default=False,
        help="Print manager/machine actions for a few early steps.",
    )
    parser.add_argument(
        "--debug_action_steps",
        type=int,
        default=5,
        help="Number of early steps to print when debug_actions is enabled.",
    )
    parser.add_argument(
        "--shared_machine_policy",
        dest="shared_machine_policy",
        action="store_true",
        default=True,
        help="Share one policy among all machine agents (agents 1..num_lines).",
    )
    parser.add_argument(
        "--no_shared_machine_policy",
        dest="shared_machine_policy",
        action="store_false",
        help="Disable shared machine policy and use one policy per machine agent.",
    )

    initial_args = parser.parse_known_args(args)[0]
    config_path = initial_args.bosch_config
    default_cfg = PROJECT_ROOT / "configs" / "bosch" / "default.json"
    if config_path is None and default_cfg.is_file():
        config_path = str(default_cfg)

    cfg = _load_bosch_config(config_path) if config_path else {}

    scenario_keys = {
        "num_lines",
        "num_products",
        "num_periods",
        "manager_max_horizon",
        "max_actions_per_period",
        "lookahead_days",
        "capacity_per_line",
        "processing_time_matrix",
        "production_cost_matrix",
        "setup_cost_matrix",
        "setup_time_matrix",
        "first_setup_cost",
        "first_setup_time",
        "hazard_rate",
        "pm_cost",
        "cm_cost",
        "pm_time",
        "cm_time",
        "eligibility_matrix",
        "demand_profile",
    }
    if cfg:
        missing = sorted([k for k in scenario_keys if k not in cfg])
        if missing:
            print(
                "[BOSCH config] Missing keys in config (defaults/auto-gen will be used): "
                + ", ".join(missing)
            )

    auto_filled = []
    if cfg.get("auto_generate"):
        seed = cfg.get("seed", initial_args.seed)
        cfg, auto_filled = _auto_fill_bosch_config(cfg, seed=seed)
        if auto_filled:
            print(
                "[BOSCH config] Auto-generated: " + ", ".join(sorted(auto_filled))
            )
        if config_path:
            _save_bosch_config(config_path, cfg)
            print(f"[BOSCH config] Wrote filled config to: {config_path}")
    if cfg:
        known_dests = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in known_dests})

    all_args = parser.parse_known_args(args)[0]
    if cfg:
        for key, val in cfg.items():
            if not hasattr(all_args, key):
                setattr(all_args, key, val)
    all_args.bosch_config = config_path
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # macOS + spawn + cloudpickle can crash when pickling env fns.
    # Prefer fork when using multi-process rollouts.
    if all_args.n_rollout_threads > 1 and platform.system() == "Darwin":
        try:
            mp.set_start_method("fork")
            print("BOSCH: using multiprocessing start_method='fork' on macOS.")
        except RuntimeError:
            # Start method already set by another library.
            pass

    # Fixed env name for this problem
    all_args.env_name = "BOSCH"

    # For this hierarchical setup we use RMAPPo with separated policies per agent.
    if all_args.algorithm_name == "rmappo":
        print(
            "Using rmappo; setting use_recurrent_policy=True and use_naive_recurrent_policy=False"
        )
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "Using mappo; setting use_recurrent_policy=False and use_naive_recurrent_policy=False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError("Only rmappo/mappo are supported for BOSCH.")

    # The manager (agent 0) and machines use different action spaces, so
    # global share_policy cannot be used.
    if all_args.share_policy:
        print("BOSCH env requires non-shared policies; overriding share_policy=False.")
        all_args.share_policy = False

    if all_args.shared_machine_policy:
        print("BOSCH: enabling shared machine policy across agents 1..num_lines.")
    else:
        print("BOSCH: using separate policy per machine agent.")

    print("BOSCH: using dense machine rewards.")

    # Centralized critic by default
    all_args.use_centralized_V = True
    # Manager should be trained only on manager decision steps via active_masks.
    all_args.use_policy_active_masks = True
    all_args.use_value_active_masks = True

    # Align episode_length (max timesteps) with micro-step horizon
    # 1 manager step + max_actions_per_period machine steps per period
    all_args.episode_length = all_args.num_periods * (
        all_args.max_actions_per_period + 1
    )

    # cuda setup
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb / logging
    if all_args.use_wandb and wandb is None:
        print("wandb is not installed; falling back to tensorboard logging.")
        all_args.use_wandb = False

    run = None
    if all_args.use_wandb:
        try:
            run = wandb.init(
                config=all_args,
                project=all_args.env_name,
                notes=socket.gethostname(),
                name=str(all_args.algorithm_name)
                + "_"
                + str(all_args.experiment_name)
                + "_seed"
                + str(all_args.seed),
                group="bosch_parallel_lines",
                dir=str(run_dir),
                job_type="training",
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"wandb init failed ({e}); falling back to tensorboard logging.")
            all_args.use_wandb = False

    if not all_args.use_wandb:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    if setproctitle is not None:
        setproctitle.setproctitle(
            str(all_args.algorithm_name)
            + "-"
            + str(all_args.env_name)
            + "-"
            + str(all_args.experiment_name)
            + "@"
            + str(all_args.user_name)
        )

    # seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = 1 + all_args.num_lines

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # Always use separated runner (one policy per agent)
    from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()

    # cleanup
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        if hasattr(runner.writter, "export_scalars_to_json"):
            runner.writter.export_scalars_to_json(
                str(runner.log_dir + "/summary.json")
            )
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
