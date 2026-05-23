from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    area_width: float = 100.0
    area_height: float = 100.0
    num_nodes: int = 100
    rounds: int = 100
    seed: int = 42
    init_energy: float = 0.5
    cluster_head_probability: float = 0.05
    packet_size: int = 4000
    base_station: tuple[float, float] = (50.0, 120.0)
    e_elec: float = 50e-9
    epsilon_fs: float = 10e-12
    epsilon_mp: float = 0.0013e-12
    e_da: float = 5e-9

    @property
    def epoch_length(self) -> int:
        return max(1, round(1.0 / self.cluster_head_probability))

    @property
    def expected_cluster_heads(self) -> float:
        return self.num_nodes * self.cluster_head_probability

    @property
    def distance_threshold(self) -> float:
        return math.sqrt(self.epsilon_fs / self.epsilon_mp)


@dataclass
class Node:
    id: int
    x: float
    y: float
    energy: float
    alive: bool = True
    is_cluster_head: bool = False
    cluster_id: int | None = None
    last_cluster_head_round: int = -10_000

    def distance_to(self, point: tuple[float, float]) -> float:
        return math.hypot(self.x - point[0], self.y - point[1])

    def distance_to_node(self, other: "Node") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def consume(self, amount: float) -> None:
        if not self.alive:
            return
        self.energy = max(0.0, self.energy - amount)
        if self.energy <= 0.0:
            self.alive = False
            self.is_cluster_head = False
            self.cluster_id = None

    def clone(self) -> "Node":
        return Node(
            id=self.id,
            x=self.x,
            y=self.y,
            energy=self.energy,
            alive=self.alive,
            is_cluster_head=self.is_cluster_head,
            cluster_id=self.cluster_id,
            last_cluster_head_round=self.last_cluster_head_round,
        )


@dataclass
class RoundRecord:
    round_number: int
    alive_nodes: int
    dead_nodes: int
    total_energy: float
    cluster_heads: int = 0
    avg_cluster_size: float = 0.0
    max_cluster_size: int = 0
    tx_packets_to_base: int = 0
    tx_packets_to_heads: int = 0
    protocol: str = "LEACH"


@dataclass
class SimulationResult:
    protocol: str
    nodes: list[Node]
    records: list[RoundRecord]
    lifecycle: dict[str, int | None]
    snapshots: dict[int, list[Node]] = field(default_factory=dict)


def tx_energy(config: SimulationConfig, distance: float) -> float:
    amplifier_energy = (
        config.epsilon_fs * distance * distance
        if distance < config.distance_threshold
        else config.epsilon_mp * distance**4
    )
    return config.packet_size * config.e_elec + config.packet_size * amplifier_energy


def rx_energy(config: SimulationConfig) -> float:
    return config.packet_size * config.e_elec


def aggregation_energy(config: SimulationConfig) -> float:
    return config.packet_size * config.e_da


def initialize_nodes(config: SimulationConfig) -> list[Node]:
    rng = np.random.default_rng(config.seed)
    xs = rng.uniform(0.0, config.area_width, size=config.num_nodes)
    ys = rng.uniform(0.0, config.area_height, size=config.num_nodes)
    return [
        Node(id=i, x=float(xs[i]), y=float(ys[i]), energy=config.init_energy)
        for i in range(config.num_nodes)
    ]


def clone_nodes(nodes: Iterable[Node]) -> list[Node]:
    return [node.clone() for node in nodes]


def living_nodes(nodes: Iterable[Node]) -> list[Node]:
    return [node for node in nodes if node.alive]


def reset_round_state(nodes: Iterable[Node]) -> None:
    for node in nodes:
        node.is_cluster_head = False
        node.cluster_id = None


def elect_cluster_heads(
    nodes: list[Node],
    round_index: int,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> list[Node]:
    position = round_index % config.epoch_length
    denominator = 1.0 - config.cluster_head_probability * position
    threshold = config.cluster_head_probability / denominator
    cluster_heads: list[Node] = []

    for node in living_nodes(nodes):
        eligible = round_index - node.last_cluster_head_round >= config.epoch_length
        if eligible and rng.random() < threshold:
            node.is_cluster_head = True
            node.cluster_id = node.id
            node.last_cluster_head_round = round_index
            cluster_heads.append(node)

    if not cluster_heads:
        candidates = living_nodes(nodes)
        if candidates:
            fallback = rng.choice(candidates)
            fallback.is_cluster_head = True
            fallback.cluster_id = fallback.id
            fallback.last_cluster_head_round = round_index
            cluster_heads.append(fallback)

    return cluster_heads


def assign_clusters(nodes: list[Node], cluster_heads: list[Node]) -> dict[int, list[Node]]:
    clusters: dict[int, list[Node]] = {head.id: [] for head in cluster_heads}
    if not cluster_heads:
        return clusters

    for node in living_nodes(nodes):
        if node.is_cluster_head:
            node.cluster_id = node.id
            continue
        nearest_head = min(cluster_heads, key=node.distance_to_node)
        node.cluster_id = nearest_head.id
        clusters[nearest_head.id].append(node)

    return clusters


def simulate_leach_round(
    nodes: list[Node],
    round_index: int,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[list[Node], dict[int, list[Node]]]:
    reset_round_state(nodes)
    cluster_heads = elect_cluster_heads(nodes, round_index, config, rng)
    clusters = assign_clusters(nodes, cluster_heads)

    head_by_id = {head.id: head for head in cluster_heads}
    for member_list in clusters.values():
        for node in member_list:
            head = head_by_id.get(node.cluster_id)
            if head is None or not head.alive:
                continue
            node.consume(tx_energy(config, node.distance_to_node(head)))
            if head.alive:
                head.consume(rx_energy(config))

    for head in cluster_heads:
        if not head.alive:
            continue
        member_count = len(clusters.get(head.id, []))
        if member_count:
            head.consume(aggregation_energy(config) * member_count)
        if head.alive:
            head.consume(tx_energy(config, head.distance_to(config.base_station)))

    return cluster_heads, clusters


def simulate_direct_round(nodes: list[Node], config: SimulationConfig) -> None:
    reset_round_state(nodes)
    for node in living_nodes(nodes):
        node.consume(tx_energy(config, node.distance_to(config.base_station)))


def make_lifecycle() -> dict[str, int | None]:
    return {
        "first_dead_round": None,
        "half_dead_round": None,
        "all_dead_round": None,
    }


def update_lifecycle(
    lifecycle: dict[str, int | None],
    record: RoundRecord,
    total_nodes: int,
) -> None:
    if record.dead_nodes >= 1 and lifecycle["first_dead_round"] is None:
        lifecycle["first_dead_round"] = record.round_number
    if record.dead_nodes >= math.ceil(total_nodes / 2) and lifecycle["half_dead_round"] is None:
        lifecycle["half_dead_round"] = record.round_number
    if record.alive_nodes == 0 and lifecycle["all_dead_round"] is None:
        lifecycle["all_dead_round"] = record.round_number


def record_leach_round(
    nodes: list[Node],
    round_number: int,
    cluster_heads: list[Node],
    clusters: dict[int, list[Node]],
) -> RoundRecord:
    alive_count = len(living_nodes(nodes))
    cluster_sizes = [len(members) + 1 for members in clusters.values()]
    return RoundRecord(
        round_number=round_number,
        alive_nodes=alive_count,
        dead_nodes=len(nodes) - alive_count,
        total_energy=sum(node.energy for node in nodes),
        cluster_heads=sum(1 for head in cluster_heads if head.alive),
        avg_cluster_size=float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        max_cluster_size=max(cluster_sizes, default=0),
        tx_packets_to_base=sum(1 for head in cluster_heads if head.alive),
        tx_packets_to_heads=sum(len(members) for members in clusters.values()),
        protocol="LEACH",
    )


def record_direct_round(nodes: list[Node], round_number: int) -> RoundRecord:
    alive_count = len(living_nodes(nodes))
    return RoundRecord(
        round_number=round_number,
        alive_nodes=alive_count,
        dead_nodes=len(nodes) - alive_count,
        total_energy=sum(node.energy for node in nodes),
        tx_packets_to_base=alive_count,
        protocol="Direct",
    )


def run_leach(config: SimulationConfig, initial_nodes: list[Node]) -> SimulationResult:
    rng = np.random.default_rng(config.seed + 1)
    nodes = clone_nodes(initial_nodes)
    records: list[RoundRecord] = []
    lifecycle = make_lifecycle()
    snapshots: dict[int, list[Node]] = {}
    snapshot_rounds = {1, min(10, config.rounds), max(1, config.rounds // 2), config.rounds}

    for round_index in range(config.rounds):
        if not living_nodes(nodes):
            lifecycle["all_dead_round"] = lifecycle["all_dead_round"] or round_index
            break

        round_number = round_index + 1
        cluster_heads, clusters = simulate_leach_round(nodes, round_index, config, rng)
        record = record_leach_round(nodes, round_number, cluster_heads, clusters)
        records.append(record)
        update_lifecycle(lifecycle, record, config.num_nodes)

        if round_number in snapshot_rounds:
            snapshots[round_number] = clone_nodes(nodes)
        if record.alive_nodes == 0:
            break

    for round_number in sorted(snapshot_rounds):
        if round_number <= config.rounds and round_number not in snapshots and records:
            snapshots[round_number] = clone_nodes(nodes)

    return SimulationResult("LEACH", nodes, records, lifecycle, snapshots)


def run_direct(config: SimulationConfig, initial_nodes: list[Node]) -> SimulationResult:
    nodes = clone_nodes(initial_nodes)
    records: list[RoundRecord] = []
    lifecycle = make_lifecycle()

    for round_index in range(config.rounds):
        if not living_nodes(nodes):
            lifecycle["all_dead_round"] = lifecycle["all_dead_round"] or round_index
            break

        round_number = round_index + 1
        simulate_direct_round(nodes, config)
        record = record_direct_round(nodes, round_number)
        records.append(record)
        update_lifecycle(lifecycle, record, config.num_nodes)
        if record.alive_nodes == 0:
            break

    return SimulationResult("Direct", nodes, records, lifecycle)


def prepare_pyplot(show: bool):
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style_network_axes(ax, title: str, config: SimulationConfig) -> None:
    ax.set_title(title)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_xlim(-5, config.area_width + 5)
    ax.set_ylim(-5, max(config.area_height, config.base_station[1]) + 10)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)


def save_or_show(fig, path: Path, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        fig.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)


def plot_initial_distribution(
    nodes: list[Node],
    config: SimulationConfig,
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter([node.x for node in nodes], [node.y for node in nodes], c="#2563eb", s=28, label="Sensor node")
    ax.scatter(*config.base_station, c="#111827", marker="*", s=190, label="Base station")
    style_network_axes(ax, "Initial WSN Node Distribution", config)
    ax.legend(loc="upper right")
    save_or_show(fig, output_dir / "initial_distribution.png", show)


def plot_cluster_snapshot(
    nodes: list[Node],
    round_number: int,
    config: SimulationConfig,
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 7))
    heads = [node for node in nodes if node.alive and node.is_cluster_head]
    normal_nodes = [node for node in nodes if node.alive and not node.is_cluster_head]
    dead_nodes = [node for node in nodes if not node.alive]
    head_by_id = {head.id: head for head in heads}

    for node in normal_nodes:
        head = head_by_id.get(node.cluster_id)
        if head is not None:
            ax.plot([node.x, head.x], [node.y, head.y], color="#94a3b8", linestyle="--", linewidth=0.7, alpha=0.45)

    for head in heads:
        ax.plot([head.x, config.base_station[0]], [head.y, config.base_station[1]], color="#dc2626", linewidth=0.9, alpha=0.55)

    if normal_nodes:
        ax.scatter([node.x for node in normal_nodes], [node.y for node in normal_nodes], c="#2563eb", s=28, label="Normal node")
    if heads:
        ax.scatter([node.x for node in heads], [node.y for node in heads], c="#dc2626", marker="^", s=78, label="Cluster head")
    if dead_nodes:
        ax.scatter([node.x for node in dead_nodes], [node.y for node in dead_nodes], c="#64748b", marker="x", s=36, label="Dead node")
    ax.scatter(*config.base_station, c="#111827", marker="*", s=190, label="Base station")

    style_network_axes(ax, f"LEACH Clustering Result - Round {round_number}", config)
    ax.legend(loc="upper right")
    save_or_show(fig, output_dir / f"cluster_round_{round_number}.png", show)


def plot_alive_curve(
    leach: SimulationResult,
    direct: SimulationResult,
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r.round_number for r in leach.records], [r.alive_nodes for r in leach.records], color="#16a34a", linewidth=2, label="LEACH")
    ax.plot([r.round_number for r in direct.records], [r.alive_nodes for r in direct.records], color="#7c3aed", linewidth=2, linestyle="--", label="Direct transmission")
    ax.set_title("Alive Nodes Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Alive nodes")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "alive_nodes_curve.png", show)


def plot_energy_curve(
    leach: SimulationResult,
    direct: SimulationResult,
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r.round_number for r in leach.records], [r.total_energy for r in leach.records], color="#f97316", linewidth=2, label="LEACH")
    ax.plot([r.round_number for r in direct.records], [r.total_energy for r in direct.records], color="#7c3aed", linewidth=2, linestyle="--", label="Direct transmission")
    ax.set_title("Total Remaining Energy Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Total energy (J)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "total_energy_curve.png", show)


def plot_cluster_head_curve(
    leach: SimulationResult,
    config: SimulationConfig,
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = [r.round_number for r in leach.records]
    counts = [r.cluster_heads for r in leach.records]
    ax.plot(rounds, counts, color="#dc2626", linewidth=1.8)
    ax.axhline(config.expected_cluster_heads, color="#111827", linewidth=1.1, linestyle="--", label="Expected value")
    ax.set_title("Cluster Head Count Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cluster heads")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "cluster_head_count_curve.png", show)


def plot_energy_comparison_bar(
    leach: SimulationResult,
    direct: SimulationResult,
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(7, 5))
    leach_final = leach.records[-1].total_energy if leach.records else 0.0
    direct_final = direct.records[-1].total_energy if direct.records else 0.0
    bars = ax.bar(["LEACH", "Direct"], [leach_final, direct_final], color=["#16a34a", "#7c3aed"])
    ax.set_title("Final Remaining Energy Comparison")
    ax.set_ylabel("Total energy (J)")
    ax.grid(True, axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom")
    save_or_show(fig, output_dir / "energy_comparison.png", show)


def write_records_csv(result: SimulationResult, output_dir: Path) -> None:
    path = output_dir / f"{result.protocol.lower()}_round_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "protocol",
            "round",
            "alive_nodes",
            "dead_nodes",
            "total_energy_j",
            "cluster_heads",
            "avg_cluster_size",
            "max_cluster_size",
            "tx_packets_to_base",
            "tx_packets_to_heads",
        ])
        for record in result.records:
            writer.writerow([
                record.protocol,
                record.round_number,
                record.alive_nodes,
                record.dead_nodes,
                f"{record.total_energy:.8f}",
                record.cluster_heads,
                f"{record.avg_cluster_size:.4f}",
                record.max_cluster_size,
                record.tx_packets_to_base,
                record.tx_packets_to_heads,
            ])


def write_combined_summary_csv(leach: SimulationResult, direct: SimulationResult, output_dir: Path) -> None:
    with (output_dir / "simulation_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "protocol",
            "round",
            "alive_nodes",
            "dead_nodes",
            "total_energy_j",
            "cluster_heads",
            "avg_cluster_size",
            "max_cluster_size",
            "tx_packets_to_base",
            "tx_packets_to_heads",
        ])
        for result in (leach, direct):
            for record in result.records:
                writer.writerow([
                    record.protocol,
                    record.round_number,
                    record.alive_nodes,
                    record.dead_nodes,
                    f"{record.total_energy:.8f}",
                    record.cluster_heads,
                    f"{record.avg_cluster_size:.4f}",
                    record.max_cluster_size,
                    record.tx_packets_to_base,
                    record.tx_packets_to_heads,
                ])


def format_lifecycle(value: int | None) -> str:
    return "未发生" if value is None else f"第 {value} 轮"


def safe_final_record(result: SimulationResult) -> RoundRecord:
    if result.records:
        return result.records[-1]
    return RoundRecord(0, 0, 0, 0.0, protocol=result.protocol)


def write_experiment_report(
    config: SimulationConfig,
    output_dir: Path,
    leach: SimulationResult,
    direct: SimulationResult,
) -> None:
    leach_final = safe_final_record(leach)
    direct_final = safe_final_record(direct)
    leach_avg_heads = float(np.mean([r.cluster_heads for r in leach.records])) if leach.records else 0.0
    energy_saved = leach_final.total_energy - direct_final.total_energy
    relative_saved = energy_saved / (config.num_nodes * config.init_energy) * 100.0

    report = f"""# WSN 节点分簇聚合模拟实验报告

## 一、实验目的

1. 了解无线传感器网络 WSN 的基本组成与通信特点。
2. 理解 LEACH 分簇聚合算法的簇头轮换、节点入簇和数据聚合思想。
3. 使用 Python 3.13 编写多轮 WSN 节点分簇聚合仿真程序。
4. 通过图表分析节点存活数量、网络剩余能量和簇头数量变化，并比较 LEACH 与普通直接传输的能耗差异。

## 二、实验环境

| 环境项目 | 内容 |
| --- | --- |
| 编程语言 | Python 3.13 |
| 第三方库 | numpy, matplotlib |
| 程序文件 | `leach_simulation.py` |
| 输出目录 | `{output_dir}` |

## 三、实验原理

WSN 是由大量低功耗传感器节点组成的无线网络，节点通常能量有限，直接向远端基站发送数据会快速消耗电量。LEACH 协议通过随机轮换簇头，将网络划分为多个簇：普通节点把数据发送给最近簇头，簇头接收并聚合数据后再发送到基站。这样可以减少长距离发送数据包的数量，并避免固定簇头长期承担高能耗任务。

LEACH 簇头阈值函数为：

```text
T(n) = p / (1 - p * (r mod (1 / p))), n 属于最近 1/p 轮未当过簇头的集合 G
T(n) = 0,                             n 不属于 G
```

其中 `p` 为期望簇头比例，`r` 为当前轮次。程序还设置了兜底机制：若某轮随机选举没有产生簇头，则从存活节点中随机选择一个簇头，保证本轮分簇能够继续进行。

能量模型采用一阶无线电模型。短距离使用自由空间模型，长距离使用多径衰落模型，距离阈值为 `d0 = sqrt(epsilon_fs / epsilon_mp)`：

```text
发送 k bit 到距离 d：
  d < d0  时，E_tx = k * E_elec + k * epsilon_fs * d^2
  d >= d0 时，E_tx = k * E_elec + k * epsilon_mp * d^4
接收 k bit：E_rx = k * E_elec
数据聚合：  E_DA = k * E_da
```

## 四、程序设计

| 模块 | 说明 |
| --- | --- |
| 参数配置 | `SimulationConfig` 保存区域、节点数量、轮数、能量模型和基站位置 |
| 节点模型 | `Node` 保存编号、坐标、剩余能量、存活状态、簇头状态和所属簇 |
| LEACH 仿真 | 每轮完成簇头选举、最近簇头入簇、簇内传输、簇头聚合和基站传输 |
| 直接传输仿真 | 使用相同初始节点，每轮所有存活节点直接向基站发送数据 |
| 统计模块 | 记录存活节点数、死亡节点数、总能量、簇头数、平均簇规模和发送包数量 |
| 可视化模块 | 输出节点分布图、分簇快照、存活曲线、能量曲线、簇头数量曲线和能量对比图 |

## 五、实验参数

| 参数 | 取值 |
| --- | --- |
| 仿真区域 | {config.area_width:.0f} m x {config.area_height:.0f} m |
| 节点数量 | {config.num_nodes} |
| 仿真轮数 | {config.rounds} |
| 随机种子 | {config.seed} |
| 初始能量 | {config.init_energy} J |
| 期望簇头比例 | {config.cluster_head_probability} |
| 理论平均簇头数 | {config.expected_cluster_heads:.2f} |
| 数据包大小 | {config.packet_size} bit |
| 基站位置 | {config.base_station} |
| E_elec | {config.e_elec:.2e} J/bit |
| epsilon_fs | {config.epsilon_fs:.2e} J/bit/m^2 |
| epsilon_mp | {config.epsilon_mp:.2e} J/bit/m^4 |
| 距离阈值 d0 | {config.distance_threshold:.2f} m |
| E_da | {config.e_da:.2e} J/bit |

## 六、实验结果

程序运行后生成以下主要结果文件：

| 文件 | 说明 |
| --- | --- |
| `initial_distribution.png` | 初始节点随机分布图 |
| `cluster_round_1.png` | 第 1 轮 LEACH 分簇结果 |
| `cluster_round_{min(10, config.rounds)}.png` | 第 {min(10, config.rounds)} 轮 LEACH 分簇结果 |
| `cluster_round_{max(1, config.rounds // 2)}.png` | 中间轮次 LEACH 分簇结果 |
| `cluster_round_{config.rounds}.png` | 最后一轮 LEACH 分簇结果 |
| `alive_nodes_curve.png` | LEACH 与直接传输的存活节点数量对比 |
| `total_energy_curve.png` | LEACH 与直接传输的网络总剩余能量对比 |
| `cluster_head_count_curve.png` | LEACH 每轮簇头数量变化 |
| `energy_comparison.png` | 最终剩余能量柱状对比 |
| `simulation_summary.csv` | 两种协议的逐轮统计数据 |

主要结果图如下：

![初始节点分布](outputs/initial_distribution.png)

![第 1 轮 LEACH 分簇结果](outputs/cluster_round_1.png)

![第 {max(1, config.rounds // 2)} 轮 LEACH 分簇结果](outputs/cluster_round_{max(1, config.rounds // 2)}.png)

![存活节点数量变化](outputs/alive_nodes_curve.png)

![网络总剩余能量变化](outputs/total_energy_curve.png)

![簇头数量变化](outputs/cluster_head_count_curve.png)

![最终剩余能量对比](outputs/energy_comparison.png)

本次 LEACH 仿真实际完成 {leach_final.round_number} 轮，最终存活节点数为 {leach_final.alive_nodes}，最终网络总剩余能量为 {leach_final.total_energy:.6f} J。直接传输最终存活节点数为 {direct_final.alive_nodes}，最终网络总剩余能量为 {direct_final.total_energy:.6f} J。LEACH 比直接传输多保留 {energy_saved:.6f} J 能量，约占初始总能量的 {relative_saved:.2f}%。

| 指标 | LEACH | 直接传输 |
| --- | --- | --- |
| 最终存活节点数 | {leach_final.alive_nodes} | {direct_final.alive_nodes} |
| 最终死亡节点数 | {leach_final.dead_nodes} | {direct_final.dead_nodes} |
| 最终总剩余能量 | {leach_final.total_energy:.6f} J | {direct_final.total_energy:.6f} J |
| 第一个节点死亡 | {format_lifecycle(leach.lifecycle["first_dead_round"])} | {format_lifecycle(direct.lifecycle["first_dead_round"])} |
| 半数节点死亡 | {format_lifecycle(leach.lifecycle["half_dead_round"])} | {format_lifecycle(direct.lifecycle["half_dead_round"])} |
| 全部节点死亡 | {format_lifecycle(leach.lifecycle["all_dead_round"])} | {format_lifecycle(direct.lifecycle["all_dead_round"])} |

LEACH 每轮平均簇头数量为 {leach_avg_heads:.2f}，与理论期望值 {config.expected_cluster_heads:.2f} 接近。由于簇头选举包含随机性，不同轮次簇头数量会围绕期望值上下波动。

## 七、结果分析

从分簇图可以看到，普通节点会连接到距离最近的簇头，簇头再与基站通信，形成“普通节点到簇头、簇头到基站”的两级传输结构。不同轮次的红色簇头位置不同，说明 LEACH 通过轮换机制分摊了高能耗任务。

从能量曲线可以看到，网络总能量随通信轮次增加持续下降。直接传输中每个节点都需要远距离发送到基站，整体能量下降更快；LEACH 让多数节点只进行短距离簇内发送，只有少量簇头向基站发送聚合数据，因此在相同轮数后保留了更多能量。

从簇头数量曲线可以看到，簇头数并不是固定值，而是在随机阈值控制下接近期望簇头比例。若某轮没有自然产生簇头，程序会随机补选一个存活节点，避免网络无法分簇。

## 八、实验总结

本实验完成了 WSN 节点随机部署、LEACH 簇头选举、最近簇头入簇、数据聚合传输、能量消耗统计和结果可视化。相比简单静态散点图，本程序实现了多轮动态仿真，并增加了与直接传输方式的对比，更直观地展示了 LEACH 协议在降低长距离通信开销、均衡节点能耗方面的作用。
"""
    Path("Experiment_Report.md").write_text(report, encoding="utf-8")


def validate_result(result: SimulationResult) -> None:
    for previous, current in zip(result.records, result.records[1:]):
        if current.alive_nodes > previous.alive_nodes:
            raise RuntimeError(f"{result.protocol}: alive node count increased unexpectedly.")
        if current.total_energy > previous.total_energy + 1e-12:
            raise RuntimeError(f"{result.protocol}: total energy increased unexpectedly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEACH simulation for wireless sensor networks.")
    parser.add_argument("--nodes", type=int, default=100, help="Number of sensor nodes.")
    parser.add_argument("--rounds", type=int, default=100, help="Number of simulation rounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated charts and CSV summary.")
    parser.add_argument("--show", action="store_true", help="Display figures in addition to saving them.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimulationConfig:
    if args.nodes <= 0:
        raise ValueError("--nodes must be greater than 0.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than 0.")
    return SimulationConfig(num_nodes=args.nodes, rounds=args.rounds, seed=args.seed)


def configure_output_environment(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir = Path(args.output_dir)
    configure_output_environment(output_dir)

    initial_nodes = initialize_nodes(config)
    leach = run_leach(config, initial_nodes)
    direct = run_direct(config, initial_nodes)
    validate_result(leach)
    validate_result(direct)

    plot_initial_distribution(initial_nodes, config, output_dir, args.show)
    for round_number in sorted(leach.snapshots):
        plot_cluster_snapshot(leach.snapshots[round_number], round_number, config, output_dir, args.show)
    plot_alive_curve(leach, direct, output_dir, args.show)
    plot_energy_curve(leach, direct, output_dir, args.show)
    plot_cluster_head_curve(leach, config, output_dir, args.show)
    plot_energy_comparison_bar(leach, direct, output_dir, args.show)
    write_records_csv(leach, output_dir)
    write_records_csv(direct, output_dir)
    write_combined_summary_csv(leach, direct, output_dir)
    write_experiment_report(config, output_dir, leach, direct)

    leach_final = safe_final_record(leach)
    direct_final = safe_final_record(direct)
    print("WSN simulation completed.")
    print(f"Rounds simulated: {leach_final.round_number}")
    print(f"LEACH alive nodes: {leach_final.alive_nodes}/{config.num_nodes}")
    print(f"LEACH total remaining energy: {leach_final.total_energy:.6f} J")
    print(f"Direct total remaining energy: {direct_final.total_energy:.6f} J")
    print(f"Output directory: {output_dir}")
    print("Report: Experiment_Report.md")


if __name__ == "__main__":
    main()
