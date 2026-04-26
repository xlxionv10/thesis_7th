import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from onpolicy.utils.multi_discrete import MultiDiscrete


class BoschEnv(object):
    """
    Parallel-line production + maintenance environment for multi-agent RL.

    Agents:
        agent 0 : lot sizing & line allocation agent (global)
        agents 1-6 : machine agents, one per line
    """

    def __init__(self, args):
        # Core configuration (with sensible defaults if attributes are missing)
        self.num_lines = getattr(args, "num_lines", 6)
        self.num_products = getattr(args, "num_products", 3)
        # Number of decision periods (planning horizon)
        num_periods = getattr(args, "num_periods", None)
        if num_periods is None:
            num_periods = getattr(args, "episode_length", 24)
        self.num_periods = int(num_periods)

        # Maximum number of machine actions (micro-steps) per period
        self.max_actions_per_period = int(
            getattr(args, "max_actions_per_period", 8)
        )

        # Capacity and demand parameters
        self.capacity_per_line = self._get_array_arg(
            args, "capacity_per_line", self.num_lines, default=100.0
        )
        # Manager horizon (days of demand to cover per line)
        self.manager_max_horizon = int(
            getattr(args, "manager_max_horizon", 7)
        )

        # Cost parameters
        self.holding_cost = float(getattr(args, "holding_cost", 1.0))
        self.backlog_cost = float(getattr(args, "backlog_cost", 10.0))
        self.per_product_backlog_penalty = float(
            getattr(args, "per_product_backlog_penalty", 500.0)
        )
        self.production_cost = float(getattr(args, "production_cost", 1.0))
        self.setup_cost = float(getattr(args, "setup_cost", 2.0))
        self.pm_cost = self._get_array_arg(
            args, "pm_cost", self.num_lines, default=20.0
        )
        self.cm_cost = self._get_array_arg(
            args, "cm_cost", self.num_lines, default=40.0
        )
        self.alpha_cost_weight = float(getattr(args, "alpha_cost_weight", 0.1))

        # Optional: share end-of-period service costs with machine agents
        # (helps align dense machine shaping with global objective).
        self.machine_service_cost_share_beta = float(
            getattr(args, "machine_service_cost_share_beta", 0.0)
        )
        self.machine_service_cost_share_mode = str(
            getattr(args, "machine_service_cost_share_mode", "assignment")
        ).strip().lower()
        self.machine_service_cost_share_include_inventory = bool(
            getattr(args, "machine_service_cost_share_include_inventory", False)
        )
        self.machine_service_cost_share_include_backlog = bool(
            getattr(args, "machine_service_cost_share_include_backlog", True)
        )

        # Optional metadata for reporting
        self.product_codes = getattr(args, "product_codes", None)
        self.line_codes = getattr(args, "line_codes", None)

        # Per-line hazard-rate style degradation (scalar or list)
        self.hazard_rate = self._get_array_arg(
            args, "hazard_rate", self.num_lines, default=1e-3
        )

        # Time-related parameters (capacity is in "hours" or generic time units)
        self.pm_time = self._get_array_arg(
            args, "pm_time", self.num_lines, default=0.0
        )
        self.cm_time = self._get_array_arg(
            args, "cm_time", self.num_lines, default=0.0
        )

        # Reward shaping options for machine agents
        self.dense_production_reward = float(
            getattr(args, "dense_production_reward", 1.0)
        )
        self.dense_setup_penalty = float(getattr(args, "dense_setup_penalty", 1.0))
        self.dense_pm_penalty = float(getattr(args, "dense_pm_penalty", 1.0))

        # Per-product processing times and mean demand (kept for compatibility)
        self.processing_time = self._get_array_arg(
            args, "processing_time", self.num_products, default=1.0
        )
        self.mean_demand = self._get_array_arg(
            args, "mean_demand", self.num_products, default=10.0
        )

        # Per-line processing time matrix (lines x products). Falls back to per-product values.
        if getattr(args, "processing_time_matrix", None) is not None:
            self.processing_time_matrix = self._get_matrix_arg(
                args,
                "processing_time_matrix",
                (self.num_lines, self.num_products),
                default=1.0,
            )
        else:
            self.processing_time_matrix = np.tile(
                self.processing_time[None, :], (self.num_lines, 1)
            ).astype(np.float32)

        # Sequence-dependent setup cost/time matrices and heterogeneous production costs
        base_setup_cost = float(getattr(args, "setup_cost", 2.0))
        base_setup_time = float(getattr(args, "setup_time", 0.0))
        self.first_setup_cost = self._get_array_arg(
            args, "first_setup_cost", self.num_lines, default=base_setup_cost
        )
        self.first_setup_time = self._get_array_arg(
            args, "first_setup_time", self.num_lines, default=base_setup_time
        )
        self.setup_cost_matrix = self._get_tensor_arg(
            args,
            "setup_cost_matrix",
            (self.num_lines, self.num_products, self.num_products),
            default=base_setup_cost,
        )
        self.setup_time_matrix = self._get_tensor_arg(
            args,
            "setup_time_matrix",
            (self.num_lines, self.num_products, self.num_products),
            default=base_setup_time,
        )

        base_prod_cost = float(getattr(args, "production_cost", 1.0))
        self.production_cost_matrix = self._get_matrix_arg(
            args,
            "production_cost_matrix",
            (self.num_lines, self.num_products),
            default=base_prod_cost,
        )

        # Product-line eligibility restrictions (1 if line can produce product, else 0)
        self.line_eligibility = self._get_matrix_arg(
            args,
            "eligibility_matrix",
            (self.num_lines, self.num_products),
            default=1.0,
        )
        self.line_eligibility = (self.line_eligibility > 0.5).astype(np.float32)

        demand_profile_raw = getattr(args, "demand_profile", None)
        if demand_profile_raw is not None:
            self.demand_profile = self._get_matrix_arg(
                args,
                "demand_profile",
                (self.num_periods, self.num_products),
                default=0.0,
            )

        # Time / episode tracking (micro-steps within periods)
        self.current_step = 0
        self.period_index = 0
        self.step_in_period = 0

        # Agents
        # agent 0 : lot sizing & allocation
        # agents 1..num_lines : machine agents
        self.num_agents = 1 + self.num_lines

        # Observation layout (same length for all agents)
        # [inventory (P),
        #  backlog (P),
        #  queue (L*P) [manager sees all line queues;
        #              machines see own line queue in its segment],
        #  coverage (P) [manager sees per-product line coverage; machines see zeros],
        #  queue_total_per_product (P) [manager only; machines see zeros],
        #  shortfall_per_product (P) [manager only; machines see zeros],
        #  demand_window (lookahead_days * P),
        #  remaining_periods (1),
        #  line_availability (L),
        #  line_setup (L one-hot over products, flattened),
        #  ages (L),
        #  local_line_id_one_hot (L),
        #  line_contention (L) [manager only; machines see zeros],
        #  product_urgency (P) [manager only; machines see zeros],
        #  line_eligibility (L*P) [manager only; machines see zeros]]
        self.lookahead_days = int(getattr(args, "lookahead_days", 5))
        self.obs_dim = (
            2 * self.num_products
            + self.num_lines * self.num_products
            + self.num_products
            + self.num_products
            + self.num_products
            + self.lookahead_days * self.num_products
            + 1
            + self.num_lines
            + self.num_lines * self.num_products
            + self.num_lines
            + self.num_lines
            + self.num_lines
            + self.num_products
            + self.num_lines * self.num_products
        )

        high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        low = -high
        self.observation_space = [
            spaces.Box(low=low, high=high, dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # Centralized observation = concatenation of all agent observations
        share_high = np.full(self.obs_dim * self.num_agents, np.inf, dtype=np.float32)
        share_low = -share_high
        self.share_observation_space = [
            spaces.Box(low=share_low, high=share_high, dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # Action spaces
        # Agent 0: MultiDiscrete per line:
        #   (primary_product, horizon_days, secondary_product)
        # primary_product  in [0, num_products-1]
        # horizon_days     in [0, manager_max_horizon]
        # secondary_product in [0, num_products]  (num_products = "none")
        lot_dims = []
        for _ in range(self.num_lines):
            lot_dims.append([0, self.num_products - 1])   # primary product
            lot_dims.append([0, self.manager_max_horizon]) # horizon
            lot_dims.append([0, self.num_products])        # secondary product (P = none)
        agent0_act = MultiDiscrete(lot_dims)

        # Machine agents: Discrete
        #   0..num_products-1 => process product
        #   num_products      => perform PM
        #   num_products + 1  => end shift
        machine_act = spaces.Discrete(self.num_products + 2)

        self.action_space = [agent0_act] + [machine_act for _ in range(self.num_lines)]

        # Internal state
        self.rng = np.random.RandomState(getattr(args, "seed", 1))

        self._build_demand_profile()
        self._reset_state()

    def _get_array_arg(self, args, name, length, default):
        """
        Helper to read scalar or comma-separated list arguments into
        a 1D float array of shape [length].
        """
        raw = getattr(args, name, None)
        if raw is None:
            return np.ones(length, dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, length)

        if arr.size != length:
            raise ValueError(
                f"Argument {name} expects length {length}, got {arr.size}."
            )
        return arr

    def _get_matrix_arg(self, args, name, shape, default):
        """
        Helper to read scalar or comma-separated list arguments into
        a 2D float array with given shape.
        """
        raw = getattr(args, name, None)
        rows, cols = shape
        if raw is None:
            return np.ones((rows, cols), dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, rows * cols)

        if arr.size != rows * cols:
            raise ValueError(
                f"Argument {name} expects {rows * cols} values, got {arr.size}."
            )
        return arr.reshape(rows, cols)

    def _get_tensor_arg(self, args, name, shape, default):
        """
        Helper to read scalar or comma-separated list arguments into
        a float tensor with given shape.
        For setup matrices, also accepts a single product×product matrix
        and tiles it across lines.
        """
        raw = getattr(args, name, None)
        total = int(np.prod(shape))
        if raw is None:
            return np.ones(shape, dtype=np.float32) * float(default)

        if isinstance(raw, (list, np.ndarray)):
            arr = np.asarray(raw, dtype=np.float32)
        else:
            parts = str(raw).split(",")
            arr = np.array([float(p) for p in parts], dtype=np.float32)

        if arr.size == 1:
            arr = np.repeat(arr, total)

        per_line = int(np.prod(shape[1:]))
        if arr.size == per_line and shape[0] > 1:
            arr = np.tile(arr, shape[0])

        if arr.size != total:
            raise ValueError(
                f"Argument {name} expects {total} values (or {per_line} to "
                f"tile across lines), got {arr.size}."
            )
        return arr.reshape(shape)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

    def reset(self):
        self._reset_state()
        self._start_new_period()
        return self._build_observations()

    def step(self, actions_env):
        """
        One environment step corresponds to a single micro-step within
        a period. Each period consists of:
            - Manager phase (step_in_period == 0): Agent 0 allocates lots.
            - Worker phases (subsequent micro-steps): machine agents act
              repeatedly until capacity or queue is exhausted.

        :param actions_env: list of per-agent one-hot / multi-one-hot actions.
        """
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        done = False

        is_manager_step = self.step_in_period == 0

        if is_manager_step:
            self._manager_step(actions_env)
        else:
            self._machines_step(actions_env, rewards)

        # Track average queue size per line across micro-steps in this period
        self.period_queue_sum += np.sum(self.queue, axis=1)
        self.period_queue_steps += 1

        # Advance micro-step counters
        self.step_in_period += 1
        self.current_step += 1

        # Decide whether to end this period
        end_of_period = (
            self.step_in_period > self.max_actions_per_period
            or self._period_effectively_over()
        )

        if end_of_period:
            day_rewards = self._end_period()
            rewards += day_rewards
            self.period_index += 1
            self.step_in_period = 0

            if self.period_index < self.num_periods:
                self._start_new_period()
            else:
                done = True

        obs = self._build_observations()
        dones = [done for _ in range(self.num_agents)]
        next_available_actions = self._build_available_actions()
        infos = []
        for agent_id in range(self.num_agents):
            info = {"available_actions": next_available_actions[agent_id].copy()}
            if agent_id == 0:
                # This flag is for next-step policy loss masking in the runner.
                info["manager_active"] = 1.0 if is_manager_step else 0.0
                if end_of_period:
                    info["period_inv_cost"] = float(self.last_inv_cost)
                    info["period_backlog_cost"] = float(self.last_backlog_cost)
                    info["period_inv_qty"] = float(self.last_inv_qty)
                    info["period_backlog_qty"] = float(self.last_backlog_qty)
                    info["period_manager_horizons"] = (
                        self.last_manager_horizons.copy()
                    )
                    info["period_manager_horizon_mean"] = float(
                        np.mean(self.last_manager_horizons)
                    )
                    info["period_index"] = int(self.last_period_index)
                    info["period_capacity_per_product"] = (
                        self.last_capacity_per_product.copy()
                    )
                    info["period_assigned_lines_per_product"] = (
                        self.last_assigned_lines_per_product.copy()
                    )
                    info["period_unmet_demand_per_product"] = (
                        self.last_unmet_demand_per_product.copy()
                    )
                    if self.product_codes is not None:
                        info["period_product_codes"] = list(self.product_codes)
                    if self.line_codes is not None:
                        info["period_line_codes"] = list(self.line_codes)
                    info["period_queue_avg_per_line"] = self.last_queue_avg_per_line.copy()
                    info["period_backlog_per_product"] = self.last_backlog_per_product.copy()
                    info["period_inventory_per_product"] = self.last_inventory_per_product.copy()
                    info["period_prod_cost"] = float(self.last_prod_cost)
                    info["period_prod_cost_per_line"] = self.last_prod_cost_per_line.copy()
                    info["period_setup_cost"] = float(self.last_setup_cost)
                    info["period_setup_cost_per_line"] = self.last_setup_cost_per_line.copy()
                    info["period_pm_cost"] = float(self.last_pm_cost)
                    info["period_cm_cost"] = float(self.last_cm_cost)
                    info["period_utilization"] = float(self.last_utilization_total)
                    info["period_utilization_per_line"] = self.last_utilization_per_line.copy()
            else:
                # Machine agents are inactive on manager micro-steps.
                info["machine_active"] = 0.0 if is_manager_step else 1.0
            infos.append(info)

        return obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_demand_profile(self):
        """
        Simple deterministic demand: constant per product per period.
        Can be replaced with real Bosch parameters or samples.
        """
        if hasattr(self, "demand_profile"):
            self.demand = self.demand_profile.astype(np.float32).copy()
            self.avg_demand_per_product = np.mean(self.demand, axis=0).astype(
                np.float32
            )
            return

        base_demand = self.mean_demand.astype(np.float32)
        self.demand = np.tile(base_demand[None, :], (self.num_periods, 1))
        self.avg_demand_per_product = np.mean(self.demand, axis=0).astype(
            np.float32
        )

    def _reset_state(self):
        # Episode-level counters
        self.current_step = 0
        self.period_index = 0
        self.step_in_period = 0

        # Inventory & backlog per product
        self.inventory = np.zeros(self.num_products, dtype=np.float32)
        self.backlog = np.zeros(self.num_products, dtype=np.float32)

        # Machine line states
        self.ages = np.zeros(self.num_lines, dtype=np.float32)
        # -1 means "no previous product"
        self.line_setup = np.full(self.num_lines, -1, dtype=np.int32)

        # Queues per line and product (assigned by manager, consumed by machines)
        self.queue = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )

        # Per-period aggregates
        self.remaining_capacity = np.zeros(self.num_lines, dtype=np.float32)
        self.line_done = np.zeros(self.num_lines, dtype=bool)
        self.period_produced_per_line = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        self.period_produced_per_product = np.zeros(
            self.num_products, dtype=np.float32
        )
        self.period_setup_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_pm_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_cm_costs = np.zeros(self.num_lines, dtype=np.float32)
        self.period_queue_sum = np.zeros(self.num_lines, dtype=np.float32)
        self.period_queue_steps = 0

        # Last-period stats for logging
        self.last_inv_cost = 0.0
        self.last_backlog_cost = 0.0
        self.last_inv_qty = 0.0
        self.last_backlog_qty = 0.0
        self.last_manager_horizons = np.zeros(self.num_lines, dtype=np.float32)
        self.last_manager_products = np.full(self.num_lines, -1, dtype=np.int32)
        self.last_capacity_per_product = np.zeros(
            self.num_products, dtype=np.float32
        )
        self.last_assigned_lines_per_product = np.zeros(
            self.num_products, dtype=np.float32
        )
        self.last_unmet_demand_per_product = np.zeros(
            self.num_products, dtype=np.float32
        )
        self.last_period_index = 0
        self.last_queue_avg_per_line = np.zeros(self.num_lines, dtype=np.float32)
        self.last_backlog_per_product = np.zeros(self.num_products, dtype=np.float32)
        self.last_inventory_per_product = np.zeros(self.num_products, dtype=np.float32)
        self.last_prod_cost = 0.0
        self.last_prod_cost_per_line = np.zeros(self.num_lines, dtype=np.float32)
        self.last_setup_cost = 0.0
        self.last_setup_cost_per_line = np.zeros(self.num_lines, dtype=np.float32)
        self.last_pm_cost = 0.0
        self.last_cm_cost = 0.0
        self.last_product_service_costs = np.zeros(
            self.num_products, dtype=np.float32
        )
        self.last_utilization_total = 0.0
        self.last_utilization_per_line = np.zeros(self.num_lines, dtype=np.float32)

    def _start_new_period(self):
        # Reset per-period capacity, flags, and aggregates, but keep
        # inventory, backlog, ages and queue.
        self.remaining_capacity[:] = self.capacity_per_line
        self.line_done[:] = False
        self.period_produced_per_line.fill(0.0)
        self.period_produced_per_product.fill(0.0)
        self.period_setup_costs.fill(0.0)
        self.period_pm_costs.fill(0.0)
        self.period_cm_costs.fill(0.0)
        self.period_queue_sum.fill(0.0)
        self.period_queue_steps = 0

    def _manager_step(self, actions_env):
        """
        Agent 0 chooses lot allocations once per period.

        The manager selects a primary (product, horizon) and an optional
        secondary product per line.  Primary allocation fills the queue
        based on demand and net requirements.  The secondary pass fills
        remaining line capacity with the secondary product, enabling
        cross-scheduling (e.g. Line 2 produces P097 then P100).
        """
        # Decode agent 0 action (primary product, horizon, secondary product)
        products, horizons, secondary_products = self._decode_agent0_action(
            actions_env[0]
        )
        self.last_manager_horizons = horizons.astype(np.float32).copy()
        self.last_manager_products = products.astype(np.int32).copy()

        # ------------------------------------------------------------------
        # PRIMARY ALLOCATION (unchanged logic)
        # ------------------------------------------------------------------
        # Add allocated lots into each line's queue based on demand horizon.
        # Backlog is shared across lines choosing the same product and
        # is split proportionally by each line's horizon demand.
        lines_by_product = {p: [] for p in range(self.num_products)}
        demand_by_line = np.zeros(self.num_lines, dtype=np.float32)
        total_demand_by_product = np.zeros(self.num_products, dtype=np.float32)
        for line_idx in range(self.num_lines):
            prod_idx = int(products[line_idx])
            horizon = int(horizons[line_idx])
            if horizon <= 0:
                continue
            if self.line_eligibility[line_idx, prod_idx] < 0.5:
                continue
            start = self.period_index
            end = min(self.period_index + horizon, self.num_periods)
            demand_sum = float(np.sum(self.demand[start:end, prod_idx]))
            if demand_sum <= 0.0 and self.backlog[prod_idx] <= 0.0:
                continue
            lines_by_product[prod_idx].append(line_idx)
            demand_by_line[line_idx] = demand_sum
            total_demand_by_product[prod_idx] += demand_sum

        net_req_by_product = np.zeros(self.num_products, dtype=np.float32)
        for prod_idx in range(self.num_products):
            if not lines_by_product[prod_idx]:
                continue
            # Credit only queue that is realistically executable under the
            # currently selected horizons and per-line setup/capacity limits.
            effective_in_progress = 0.0
            for line_idx in lines_by_product[prod_idx]:
                horizon = int(horizons[line_idx])
                if horizon <= 0:
                    continue

                proc_time = float(self.processing_time_matrix[line_idx, prod_idx])
                if proc_time <= 0.0:
                    continue

                setup_time_for_line = 0.0
                last_prod = int(self.line_setup[line_idx])
                if last_prod < 0:
                    setup_time_for_line = float(self.first_setup_time[line_idx])
                elif last_prod != prod_idx:
                    setup_time_for_line = float(
                        self.setup_time_matrix[line_idx, last_prod, prod_idx]
                    )

                cap_hours = float(self.capacity_per_line[line_idx]) * float(horizon)
                cap_hours = max(0.0, cap_hours - setup_time_for_line)
                cap_units = cap_hours / proc_time if proc_time > 0.0 else 0.0
                queued_units = float(self.queue[line_idx, prod_idx])
                effective_in_progress += min(queued_units, cap_units)

            total_need = (
                total_demand_by_product[prod_idx]
                + float(self.backlog[prod_idx])
                - float(self.inventory[prod_idx])
                - float(effective_in_progress)
            )
            net_req_by_product[prod_idx] = max(0.0, total_need)

        # Track how many capacity-hours each line consumes in the primary pass
        # so the secondary pass knows how much room is left.
        primary_hours_used = np.zeros(self.num_lines, dtype=np.float32)

        for line_idx in range(self.num_lines):
            prod_idx = int(products[line_idx])
            horizon = int(horizons[line_idx])

            if horizon <= 0:
                continue

            if self.line_eligibility[line_idx, prod_idx] < 0.5:
                continue

            start = self.period_index
            end = min(self.period_index + horizon, self.num_periods)
            demand_sum = float(np.sum(self.demand[start:end, prod_idx]))
            if demand_sum <= 0.0 and self.backlog[prod_idx] <= 0.0:
                continue

            assigned_lines = lines_by_product[prod_idx]
            if not assigned_lines:
                continue

            if total_demand_by_product[prod_idx] > 0.0:
                share_ratio = float(
                    demand_by_line[line_idx] / total_demand_by_product[prod_idx]
                )
            else:
                share_ratio = 1.0 / float(len(assigned_lines))

            # net_req_by_product already subtracts total in-progress queue for this
            # product across all lines, so do not subtract per-line queue again.
            net_new_qty = net_req_by_product[prod_idx] * share_ratio

            # Cap by line capacity over the selected horizon (net of setup time).
            setup_time_for_line = 0.0
            last_prod = int(self.line_setup[line_idx])
            if last_prod < 0:
                setup_time_for_line = float(self.first_setup_time[line_idx])
            elif last_prod != prod_idx:
                setup_time_for_line = float(
                    self.setup_time_matrix[line_idx, last_prod, prod_idx]
                )

            cap_hours = float(self.capacity_per_line[line_idx]) * float(horizon)
            proc_time = float(self.processing_time_matrix[line_idx, prod_idx])
            cap_hours = max(0.0, cap_hours - setup_time_for_line)
            cap_units = cap_hours / proc_time if proc_time > 0.0 else 0.0

            # Check how full the bucket already is!
            current_queue = float(self.queue[line_idx, prod_idx])
            available_cap = max(0.0, cap_units - current_queue)

            # Cap the addition by the ACTUAL remaining space on this line
            qty_to_add = min(net_new_qty, available_cap)

            if qty_to_add <= 0.0:
                continue

            self.queue[line_idx, prod_idx] += qty_to_add
            primary_hours_used[line_idx] += (
                qty_to_add * proc_time + setup_time_for_line
            )

        # ------------------------------------------------------------------
        # SECONDARY ALLOCATION
        # ------------------------------------------------------------------
        for line_idx in range(self.num_lines):
            sec_prod = int(secondary_products[line_idx])
            # Index == num_products means "none" — skip.
            if sec_prod >= self.num_products:
                continue
            if self.line_eligibility[line_idx, sec_prod] < 0.5:
                continue

            sec_proc_time = float(
                self.processing_time_matrix[line_idx, sec_prod]
            )
            if sec_proc_time <= 0.0:
                continue

            primary_prod = int(products[line_idx])
            horizon = int(horizons[line_idx])
            
            # Treat horizon=0 as a 1-day lookahead for the secondary product 
            # so the secondary product can still function if primary was skipped.
            effective_horizon = max(1, horizon)

            if horizon > 0 and self.line_eligibility[line_idx, primary_prod] >= 0.5:
                effective_last_prod = primary_prod
            else:
                effective_last_prod = int(self.line_setup[line_idx])

            sec_setup_time = 0.0
            if effective_last_prod < 0:
                sec_setup_time = float(self.first_setup_time[line_idx])
            elif effective_last_prod != sec_prod:
                sec_setup_time = float(
                    self.setup_time_matrix[
                        line_idx, effective_last_prod, sec_prod
                    ]
                )

            # FIX 2: Multiply total capacity by the effective horizon!
            total_cap_hours = float(self.capacity_per_line[line_idx]) * float(effective_horizon)
            
            remaining_hours = max(
                0.0,
                total_cap_hours
                - float(primary_hours_used[line_idx])
                - sec_setup_time,
            )
            if remaining_hours <= 0.0:
                continue

            remaining_units = remaining_hours / sec_proc_time

            # NEW FIX: Prevent Secondary Queue Explosion by checking the existing bucket!
            current_sec_queue = float(self.queue[line_idx, sec_prod])
            available_sec_cap = max(0.0, remaining_units - current_sec_queue)

            # FIX 1: Bound the demand sum by the effective horizon!
            start = self.period_index
            end = min(self.period_index + effective_horizon, self.num_periods)
            sec_demand_sum = float(np.sum(self.demand[start:end, sec_prod]))

            sec_need = (
                sec_demand_sum
                + float(self.backlog[sec_prod])
                - float(self.inventory[sec_prod])
                - float(np.sum(self.queue[:, sec_prod]))
            )
            sec_need = max(0.0, sec_need)
            if sec_need <= 0.0:
                continue

            qty_to_add = min(available_sec_cap, sec_need)
            if qty_to_add <= 0.0:
                continue

            self.queue[line_idx, sec_prod] += qty_to_add

    def _machines_step(self, actions_env, rewards):
        """
        Machine agents act given current queues and remaining capacity.
        """
        pm_index = self.num_products
        end_index = self.num_products + 1

        for line_idx in range(self.num_lines):
            if self.line_done[line_idx]:
                continue

            agent_id = 1 + line_idx
            a_vec = np.asarray(actions_env[agent_id], dtype=np.float32)
            act_idx = int(np.argmax(a_vec))

            # Explicit "End Shift"
            if act_idx == end_index:
                self.line_done[line_idx] = True
                continue

            # Preventive maintenance
            if act_idx == pm_index:
                pm_cost = (
                    float(self.pm_cost[line_idx])
                    if np.ndim(self.pm_cost) > 0
                    else float(self.pm_cost)
                )
                pm_time = (
                    float(self.pm_time[line_idx])
                    if np.ndim(self.pm_time) > 0
                    else float(self.pm_time)
                )
                if pm_time > 0.0 and self.remaining_capacity[line_idx] > 0.0:
                    used = min(pm_time, self.remaining_capacity[line_idx])
                    self.remaining_capacity[line_idx] -= used
                self.period_pm_costs[line_idx] += pm_cost
                self.ages[line_idx] = 0.0
                rewards[agent_id, 0] -= self.dense_pm_penalty * pm_cost
                # PM does not automatically end the shift; further actions may follow.
                continue

            # Produce a product
            if act_idx < 0 or act_idx >= self.num_products:
                # Out-of-range action is ignored; masking should prevent this.
                continue

            # Check eligibility
            if self.line_eligibility[line_idx, act_idx] < 0.5:
                # Ineligible action is ignored; masking should prevent this.
                continue

            # How much is waiting in this line's queue for this product?
            requested_qty = float(self.queue[line_idx, act_idx])
            if requested_qty <= 0.0:
                # Empty queue choice is ignored; masking should prevent this.
                continue

            last_prod = int(self.line_setup[line_idx])
            setup_time = 0.0
            setup_cost = 0.0
            if last_prod < 0:
                setup_time = float(self.first_setup_time[line_idx])
                setup_cost = float(self.first_setup_cost[line_idx])
            elif last_prod != act_idx:
                setup_time = float(self.setup_time_matrix[line_idx, last_prod, act_idx])
                setup_cost = float(self.setup_cost_matrix[line_idx, last_prod, act_idx])

            proc_time_per_unit = float(
                self.processing_time_matrix[line_idx, act_idx]
            )
            if proc_time_per_unit <= 0.0:
                # Invalid processing time choice is ignored; masking should prevent this.
                continue

            available_for_proc = self.remaining_capacity[line_idx] - setup_time
            if available_for_proc <= 0.0:
                # Not enough time for selected product; action is ignored.
                continue

            max_qty_cap = int(available_for_proc // proc_time_per_unit)
            if max_qty_cap <= 0:
                # Not enough capacity for selected product; action is ignored.
                continue

            qty = min(int(requested_qty), max_qty_cap)
            if qty <= 0:
                # Degenerate selected quantity; action is ignored.
                continue

            # Time consumed this micro-step on this line
            time_used = qty * proc_time_per_unit + setup_time
            self.remaining_capacity[line_idx] = max(
                0.0, self.remaining_capacity[line_idx] - time_used
            )

            # Update queue and production aggregates
            self.queue[line_idx, act_idx] = max(
                0.0, self.queue[line_idx, act_idx] - qty
            )
            self.period_produced_per_line[line_idx, act_idx] += qty
            self.period_produced_per_product[act_idx] += qty

            # Sequence-dependent setup cost if there was a changeover
            if setup_cost > 0.0 or setup_time > 0.0:
                self.period_setup_costs[line_idx] += setup_cost
                rewards[agent_id, 0] -= (
                    self.dense_setup_penalty * float(setup_cost)
                )

            cap = float(self.capacity_per_line[line_idx])
            if cap > 0.0:
                time_fraction = (qty * proc_time_per_unit) / cap
                rewards[agent_id, 0] += (
                    self.dense_production_reward * float(time_fraction) * cap
                )

            # Always record the last produced product (needed for next changeover calc)
            self.line_setup[line_idx] = act_idx

            # Age increases with actual runtime (excluding setup)
            delta_age = qty * proc_time_per_unit
            self.ages[line_idx] += delta_age

            # Deterministic expected CM impact based on incremental wear
            expected_failures = float(self.hazard_rate[line_idx]) * delta_age
            step_cm_cost = expected_failures * float(self.cm_cost[line_idx])
            self.period_cm_costs[line_idx] += step_cm_cost
            rewards[agent_id, 0] -= step_cm_cost

            cm_time = (
                float(self.cm_time[line_idx])
                if np.ndim(self.cm_time) > 0
                else float(self.cm_time)
            )
            step_cm_time = expected_failures * cm_time
            if step_cm_time > 0.0:
                self.remaining_capacity[line_idx] = max(
                    0.0, self.remaining_capacity[line_idx] - step_cm_time
                )

            # If no capacity remains, end shift for this line.
            if self.remaining_capacity[line_idx] <= 0.0:
                self.line_done[line_idx] = True

    def _can_process_product(self, line_idx, product_idx):
        if self.line_eligibility[line_idx, product_idx] < 0.5:
            return False

        requested_qty = float(self.queue[line_idx, product_idx])
        if requested_qty <= 0.0:
            return False

        proc_time_per_unit = float(
            self.processing_time_matrix[line_idx, product_idx]
        )
        if proc_time_per_unit <= 0.0:
            return False

        last_prod = int(self.line_setup[line_idx])
        setup_time = 0.0
        if last_prod < 0:
            setup_time = float(self.first_setup_time[line_idx])
        elif last_prod != product_idx:
            setup_time = float(self.setup_time_matrix[line_idx, last_prod, product_idx])

        available_for_proc = self.remaining_capacity[line_idx] - setup_time
        if available_for_proc <= 0.0:
            return False

        max_qty_cap = int(available_for_proc // proc_time_per_unit)
        if max_qty_cap <= 0:
            return False

        return True

    def _line_available_actions(self, line_idx):
        pm_index = self.num_products
        end_index = self.num_products + 1
        mask = np.zeros(self.num_products + 2, dtype=np.float32)

        # Machine actions are ignored in manager phase; keep a single valid action.
        if self.step_in_period == 0:
            mask[end_index] = 1.0
            return mask

        # Line already ended; only explicit end-shift is valid.
        if self.line_done[line_idx]:
            mask[end_index] = 1.0
            return mask

        # PM is always a valid control
        mask[pm_index] = 1.0
        
        # Check if there are any products we can actually process
        can_work = False
        for product_idx in range(self.num_products):
            if self._can_process_product(line_idx, product_idx):
                mask[product_idx] = 1.0
                can_work = True

        # NEW: You can only End Shift if there is no work you can do!
        if not can_work:
            mask[end_index] = 1.0

        return mask

    def _build_available_actions(self):
        # Agent 0 uses MultiDiscrete; build a per-dimension mask.
        # Layout per line: [product (P), horizon (H+1), secondary (P+1)].
        if self.num_lines > 0 and self.num_products > 0:
            masks = []
            for line_idx in range(self.num_lines):
                # Primary product: eligible products
                prod_mask = self.line_eligibility[line_idx].astype(np.float32).copy()
                if np.sum(prod_mask) <= 0.0:
                    prod_mask[:] = 1.0
                # Horizon: all horizons always valid
                horizon_mask = np.ones(self.manager_max_horizon + 1, dtype=np.float32)
                # Secondary product: eligible products + "none" (index P)
                sec_mask = np.zeros(self.num_products + 1, dtype=np.float32)
                sec_mask[:self.num_products] = (
                    self.line_eligibility[line_idx].astype(np.float32)
                )
                sec_mask[self.num_products] = 1.0  # "none" is always valid
                masks.append(prod_mask)
                masks.append(horizon_mask)
                masks.append(sec_mask)
            manager_mask = np.concatenate(masks, axis=0)
        else:
            manager_mask = np.zeros(0, dtype=np.float32)
        available_actions = [manager_mask]
        for line_idx in range(self.num_lines):
            available_actions.append(self._line_available_actions(line_idx))
        return available_actions

    def _period_effectively_over(self):
        """
        The period is effectively over if every line has either explicitly
        ended its shift, or has run out of capacity.
        """
        return bool(np.all(self.line_done | (self.remaining_capacity <= 0.0)))

    def _end_period(self):
        """
        Compute per-period costs and rewards, update inventory/backlog,
        and return a reward vector of length num_agents for this day.
        """
        self.last_period_index = self.period_index

        # Inventory / backlog cost update based on total production this period.
        inv_cost, backlog_cost, inv_costs, backlog_costs = self._update_inventory_and_backlog(
            self.period_produced_per_product
        )
        self.last_inv_cost = float(inv_cost)
        self.last_backlog_cost = float(backlog_cost)
        self.last_inv_qty = float(np.sum(self.inventory))
        self.last_backlog_qty = float(np.sum(self.backlog))
        self.last_backlog_per_product = self.backlog.astype(np.float32).copy()
        self.last_inventory_per_product = self.inventory.astype(np.float32).copy()
        self.last_unmet_demand_per_product = self.backlog.astype(np.float32).copy()

        # Eligible capacity per product (units per period)
        cap_units = np.zeros(self.num_products, dtype=np.float32)
        for p in range(self.num_products):
            mask = (self.line_eligibility[:, p] > 0.5) & (
                self.processing_time_matrix[:, p] > 0.0
            )
            if np.any(mask):
                cap_units[p] = float(
                    np.sum(
                        self.capacity_per_line[mask]
                        / self.processing_time_matrix[mask, p]
                    )
                )
        self.last_capacity_per_product = cap_units

        # Assigned lines per product (based on manager action for this period)
        assigned = np.zeros(self.num_products, dtype=np.float32)
        for line_idx in range(self.num_lines):
            prod_idx = int(self.last_manager_products[line_idx])
            if (
                prod_idx < 0
                or prod_idx >= self.num_products
                or self.last_manager_horizons[line_idx] <= 0
            ):
                continue
            if self.line_eligibility[line_idx, prod_idx] < 0.5:
                continue
            assigned[prod_idx] += 1.0
        self.last_assigned_lines_per_product = assigned

        if self.period_queue_steps > 0:
            self.last_queue_avg_per_line = (
                self.period_queue_sum / float(self.period_queue_steps)
            ).astype(np.float32)
        else:
            self.last_queue_avg_per_line = np.zeros(
                self.num_lines, dtype=np.float32
            )

        # Setup and PM costs are already accumulated per line.
        setup_cost_total = float(np.sum(self.period_setup_costs))
        pm_cost_total = float(np.sum(self.period_pm_costs))

        # Expected CM cost accumulated deterministically during the period
        expected_cm_cost_total = float(np.sum(self.period_cm_costs))

        # Heterogeneous production cost per line and product
        prod_cost_total = float(
            np.sum(self.period_produced_per_line * self.production_cost_matrix)
        )
        prod_cost_per_line = np.sum(
            self.period_produced_per_line * self.production_cost_matrix, axis=1
        ).astype(np.float32)

        # Capacity utilization (runtime / capacity)
        runtime_per_line = np.sum(
            self.period_produced_per_line * self.processing_time_matrix, axis=1
        )
        cap = self.capacity_per_line.astype(np.float32)
        util_per_line = np.zeros(self.num_lines, dtype=np.float32)
        for i in range(self.num_lines):
            if cap[i] > 0.0:
                util_per_line[i] = float(runtime_per_line[i] / cap[i])
        util_total = float(
            np.sum(runtime_per_line) / max(1e-6, float(np.sum(cap)))
        )

        # Store per-period cost breakdowns for logging
        self.last_prod_cost = prod_cost_total
        self.last_prod_cost_per_line = prod_cost_per_line
        self.last_setup_cost = setup_cost_total
        self.last_setup_cost_per_line = self.period_setup_costs.astype(np.float32).copy()
        self.last_pm_cost = pm_cost_total
        self.last_cm_cost = expected_cm_cost_total
        self.last_utilization_per_line = util_per_line
        self.last_utilization_total = util_total

        # --- 1. DIRECT COSTS ---
        manager_direct_costs = inv_cost + backlog_cost + prod_cost_total
        worker_total_direct_costs = setup_cost_total + pm_cost_total + expected_cm_cost_total

        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        # --- 2. MANAGER REWARD ---
        rewards[0, 0] = -float(manager_direct_costs) + self.alpha_cost_weight * (
            -float(worker_total_direct_costs)
        )

        # --- 3. MACHINE REWARD ---
        # Machine agents receive dense, local rewards during _machines_step().
        rewards[1:, 0] = 0.0

        # Optional: share end-of-period service costs (inventory/backlog) with machines.
        # This is a "soft team reward" that preserves dense shaping while aligning incentives.
        beta = float(self.machine_service_cost_share_beta)
        if beta > 0.0 and self.num_lines > 0:
            include_inv = bool(self.machine_service_cost_share_include_inventory)
            include_backlog = bool(self.machine_service_cost_share_include_backlog)
            if include_inv or include_backlog:
                service_costs = np.zeros(self.num_products, dtype=np.float32)
                if include_inv:
                    service_costs += inv_costs.astype(np.float32)
                if include_backlog:
                    service_costs += backlog_costs.astype(np.float32)

                mode = str(self.machine_service_cost_share_mode).lower()
                weights = np.zeros((self.num_lines, self.num_products), dtype=np.float32)

                if mode == "production":
                    weights = self.period_produced_per_line.astype(np.float32).copy()
                elif mode == "queue":
                    # Use end-of-period remaining queue as responsibility proxy.
                    weights = np.maximum(self.queue.astype(np.float32), 0.0)
                else:
                    # Default: assignment-based (manager choice this period).
                    for l in range(self.num_lines):
                        p = int(self.last_manager_products[l])
                        if 0 <= p < self.num_products and float(self.last_manager_horizons[l]) > 0.0:
                            weights[l, p] = float(self.last_manager_horizons[l])

                denom = np.sum(weights, axis=0)  # per product
                for p in range(self.num_products):
                    cost_p = float(service_costs[p])
                    if cost_p <= 0.0:
                        continue
                    d = float(denom[p])
                    if d <= 1e-8:
                        # Fall back to uniform split across all lines.
                        share = cost_p / float(self.num_lines)
                        for l in range(self.num_lines):
                            rewards[1 + l, 0] -= beta * share
                        continue
                    for l in range(self.num_lines):
                        w = float(weights[l, p])
                        if w <= 0.0:
                            continue
                        rewards[1 + l, 0] -= beta * (w / d) * cost_p

        return rewards

    def _update_inventory_and_backlog(self, produced):
        # produced is total quantity per product in this period
        t = min(self.period_index, self.num_periods - 1)
        period_demand = self.demand[t]

        inv_costs = np.zeros(self.num_products, dtype=np.float32)
        backlog_costs = np.zeros(self.num_products, dtype=np.float32)

        for p in range(self.num_products):
            total_demand = period_demand[p] + self.backlog[p]
            total_supply = self.inventory[p] + produced[p]

            if total_supply >= total_demand:
                self.inventory[p] = total_supply - total_demand
                self.backlog[p] = 0.0
            else:
                self.inventory[p] = 0.0
                self.backlog[p] = total_demand - total_supply

            inv_costs[p] = self.holding_cost * self.inventory[p]
            backlog_costs[p] = self.backlog_cost * self.backlog[p]
            if self.backlog[p] > 0:
                backlog_costs[p] += self.per_product_backlog_penalty

        self.last_product_service_costs = inv_costs + backlog_costs
        inv_cost = float(np.sum(inv_costs))
        backlog_cost = float(np.sum(backlog_costs))
        return inv_cost, backlog_cost, inv_costs, backlog_costs

    def _decode_agent0_action(self, action_vec):
        """
        Convert concatenated one-hot representation back into per-line
        (product_index, horizon_days, secondary_product) selections.
        """
        one_hot = np.asarray(action_vec, dtype=np.float32).ravel()

        num_dims = self.num_lines * 3
        block_sizes = [
            self.num_products,             # primary product
            self.manager_max_horizon + 1,  # horizon
            self.num_products + 1,         # secondary product (P = "none")
        ]

        expected_len = sum(block_sizes) * self.num_lines

        if one_hot.size != expected_len:
            raise ValueError(
                f"Agent 0 action length {one_hot.size} does not match "
                f"expected {expected_len} for "
                f"{num_dims} dims."
            )

        products = np.zeros(self.num_lines, dtype=np.float32)
        horizons = np.zeros(self.num_lines, dtype=np.float32)
        secondary_products = np.zeros(self.num_lines, dtype=np.float32)

        offset = 0
        for line_idx in range(self.num_lines):
            # primary product index
            seg = one_hot[offset : offset + block_sizes[0]]
            products[line_idx] = float(int(np.argmax(seg)))
            offset += block_sizes[0]
            # horizon days
            seg = one_hot[offset : offset + block_sizes[1]]
            horizons[line_idx] = float(int(np.argmax(seg)))
            offset += block_sizes[1]
            # secondary product index (num_products = "none")
            seg = one_hot[offset : offset + block_sizes[2]]
            secondary_products[line_idx] = float(int(np.argmax(seg)))
            offset += block_sizes[2]

        return products, horizons, secondary_products

    def _build_observations(self):
        """
        Build per-agent observations with a fixed-length layout.
        """
        remaining_periods = self.num_periods - self.period_index

        # Shared pieces
        inv = self.inventory.astype(np.float32)
        back = self.backlog.astype(np.float32)
        queue_segment_len = self.num_lines * self.num_products
        coverage = (self.queue > 0).astype(np.float32).sum(axis=0)
        queue_total = np.sum(self.queue, axis=0).astype(np.float32)
        if self.num_lines > 0:
            coverage /= float(self.num_lines)
        demand_window = np.zeros(
            (self.lookahead_days, self.num_products), dtype=np.float32
        )
        for d in range(self.lookahead_days):
            target_idx = self.period_index + d
            if target_idx < self.num_periods:
                demand_window[d] = self.demand[target_idx]

        # Contention: for each line, how many eligible products have demand this period
        contention = np.zeros(self.num_lines, dtype=np.float32)
        if self.num_periods > 0:
            t_idx = min(self.period_index, self.num_periods - 1)
            for l in range(self.num_lines):
                competing = 0
                for p in range(self.num_products):
                    if (
                        self.line_eligibility[l, p] > 0.5
                        and self.demand[t_idx, p] > 0
                    ):
                        competing += 1
                if self.num_products > 0:
                    contention[l] = float(competing) / float(self.num_products)

        # Urgency: for each product, how soon is the next non-zero demand?
        urgency = np.zeros(self.num_products, dtype=np.float32)
        for p in range(self.num_products):
            for d in range(self.lookahead_days):
                idx = self.period_index + d
                if idx < self.num_periods and self.demand[idx, p] > 0:
                    urgency[p] = 1.0 / float(d + 1)
                    break

        # Line availability = always 1 in this simplified model
        line_availability = np.ones(self.num_lines, dtype=np.float32)

        # Line setup one-hot per line over products
        line_setup_oh = np.zeros(
            (self.num_lines, self.num_products), dtype=np.float32
        )
        for l in range(self.num_lines):
            idx = int(self.line_setup[l])
            if 0 <= idx < self.num_products:
                line_setup_oh[l, idx] = 1.0
        line_setup_flat = line_setup_oh.reshape(-1)

        ages = self.ages.astype(np.float32)

        obs_all = []
        for agent_id in range(self.num_agents):
            vec = np.zeros(self.obs_dim, dtype=np.float32)
            pos = 0

            # Inventory
            vec[pos : pos + self.num_products] = inv
            pos += self.num_products

            # Backlog
            vec[pos : pos + self.num_products] = back
            pos += self.num_products

            # Queue (manager: per-line queues flattened; machines: own line queue)
            if agent_id == 0:
                queue_vec = self.queue.reshape(-1).astype(np.float32)
            else:
                queue_vec = np.zeros(queue_segment_len, dtype=np.float32)
                line_idx = agent_id - 1
                if 0 <= line_idx < self.num_lines:
                    start = line_idx * self.num_products
                    end = start + self.num_products
                    queue_vec[start:end] = self.queue[line_idx].astype(np.float32)
            vec[pos : pos + queue_segment_len] = queue_vec
            pos += queue_segment_len

            # Coverage (manager only; machines get zeros)
            if agent_id == 0:
                vec[pos : pos + self.num_products] = coverage
            pos += self.num_products

            # Queue total per product (manager only; machines get zeros)
            if agent_id == 0:
                vec[pos : pos + self.num_products] = queue_total
            pos += self.num_products

            # Shortfall per product (manager only; machines get zeros)
            if agent_id == 0:
                if demand_window.shape[0] > 0:
                    # NEW FIX: Sum the demand over the entire lookahead window!
                    demand_next = np.sum(demand_window, axis=0) 
                else:
                    demand_next = np.zeros(self.num_products, dtype=np.float32)
                shortfall = demand_next + back - inv - queue_total
                shortfall = np.maximum(shortfall, 0.0)
                vec[pos : pos + self.num_products] = shortfall.astype(np.float32)
            pos += self.num_products

            # Demand window
            window_flat = demand_window.reshape(-1)
            vec[pos : pos + window_flat.size] = window_flat
            pos += window_flat.size

            # Remaining periods
            vec[pos] = float(remaining_periods)
            pos += 1

            # Line availability
            vec[pos : pos + self.num_lines] = line_availability
            pos += self.num_lines

            # Line setup (flattened one-hot)
            vec[pos : pos + self.num_lines * self.num_products] = line_setup_flat
            pos += self.num_lines * self.num_products

            # Ages
            vec[pos : pos + self.num_lines] = ages
            pos += self.num_lines

            # Local line id one-hot (for machine agents only)
            line_id_oh = np.zeros(self.num_lines, dtype=np.float32)
            if agent_id > 0:
                line_idx = agent_id - 1
                if 0 <= line_idx < self.num_lines:
                    line_id_oh[line_idx] = 1.0
            vec[pos : pos + self.num_lines] = line_id_oh
            pos += self.num_lines

            # Line contention (manager only; machines get zeros)
            if agent_id == 0:
                vec[pos : pos + self.num_lines] = contention
            pos += self.num_lines

            # Product urgency (manager only; machines get zeros)
            if agent_id == 0:
                vec[pos : pos + self.num_products] = urgency
            pos += self.num_products

            # Line eligibility (manager only; machines get zeros)
            if agent_id == 0:
                vec[pos : pos + self.num_lines * self.num_products] = (
                    self.line_eligibility.reshape(-1).astype(np.float32)
                )
            pos += self.num_lines * self.num_products

            obs_all.append(vec)

        return np.asarray(obs_all, dtype=np.float32)

    # Optional compatibility with env_wrappers rendering
    def render(self, mode="human"):
        return None

    def close(self):
        return None
