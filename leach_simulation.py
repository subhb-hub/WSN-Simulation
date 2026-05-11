from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


AREA_WIDTH = 100.0
AREA_HEIGHT = 100.0
INIT_ENERGY = 0.5
P = 0.05
PACKET_SIZE = 4000
E_ELEC = 50e-9
E_AMP = 100e-12
E_DA = 5e-9
BASE_STATION = (50.0, 150.0)


@dataclass
class Node:
    id: int
    x: float
    y: float
    energy: float = INIT_ENERGY
    is_alive: bool = True
    is_CH: bool = False
    cluster_id: int | None = None
    last_CH_round: int = -1

    def distance_to(self, point: tuple[float, float]) -> float:
        return math.hypot(self.x - point[0], self.y - point[1])

    def distance_to_node(self, other: "Node") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def consume(self, energy_cost: float) -> None:
        if not self.is_alive:
            return
        self.energy = max(0.0, self.energy - energy_cost)
        if self.energy <= 0.0:
            self.is_alive = False
            self.is_CH = False
            self.cluster_id = None


@dataclass
class RoundRecord:
    round_number: int
    alive_nodes: int
    total_energy: float
    cluster_heads: int


def tx_energy(packet_size: int, distance: float) -> float:
    return E_ELEC * packet_size + E_AMP * packet_size * distance * distance


def rx_energy(packet_size: int) -> float:
    return E_ELEC * packet_size


def da_energy(packet_size: int) -> float:
    return E_DA * packet_size


def initialize_nodes(num_nodes: int, rng: np.random.Generator) -> list[Node]:
    xs = rng.uniform(0.0, AREA_WIDTH, size=num_nodes)
    ys = rng.uniform(0.0, AREA_HEIGHT, size=num_nodes)
    return [Node(id=i, x=float(xs[i]), y=float(ys[i])) for i in range(num_nodes)]


def reset_round_state(nodes: Iterable[Node]) -> None:
    for node in nodes:
        node.is_CH = False
        node.cluster_id = None


def alive_nodes(nodes: Iterable[Node]) -> list[Node]:
    return [node for node in nodes if node.is_alive]


def elect_cluster_heads(
    nodes: list[Node],
    round_index: int,
    p: float,
    rng: np.random.Generator,
) -> list[Node]:
    epoch = max(1, round(1.0 / p))
    threshold = p / (1.0 - p * (round_index % epoch))
    cluster_heads: list[Node] = []

    for node in alive_nodes(nodes):
        eligible = node.last_CH_round < 0 or round_index - node.last_CH_round >= epoch
        if eligible and rng.random() < threshold:
            node.is_CH = True
            node.cluster_id = node.id
            node.last_CH_round = round_index
            cluster_heads.append(node)

    if not cluster_heads:
        candidates = alive_nodes(nodes)
        if candidates:
            fallback = rng.choice(candidates)
            fallback.is_CH = True
            fallback.cluster_id = fallback.id
            fallback.last_CH_round = round_index
            cluster_heads.append(fallback)

    return cluster_heads


def assign_clusters(nodes: list[Node], cluster_heads: list[Node]) -> None:
    if not cluster_heads:
        return

    for node in alive_nodes(nodes):
        if node.is_CH:
            node.cluster_id = node.id
            continue
        nearest_head = min(cluster_heads, key=node.distance_to_node)
        node.cluster_id = nearest_head.id


def simulate_communication(
    nodes: list[Node],
    cluster_heads: list[Node],
    packet_size: int,
    base_station: tuple[float, float],
) -> None:
    head_by_id = {head.id: head for head in cluster_heads if head.is_alive}
    members_by_head: dict[int, list[Node]] = {head.id: [] for head in cluster_heads}

    for node in alive_nodes(nodes):
        if node.is_CH or node.cluster_id is None:
            continue
        head = head_by_id.get(node.cluster_id)
        if head is None:
            continue
        distance = node.distance_to_node(head)
        node.consume(tx_energy(packet_size, distance))
        if head.is_alive:
            head.consume(rx_energy(packet_size))
            if node.is_alive or node.energy == 0.0:
                members_by_head.setdefault(head.id, []).append(node)

    for head in list(cluster_heads):
        if not head.is_alive:
            continue
        member_count = len(members_by_head.get(head.id, []))
        if member_count:
            head.consume(da_energy(packet_size) * member_count)
        if head.is_alive:
            head.consume(tx_energy(packet_size, head.distance_to(base_station)))


def record_round(nodes: list[Node], round_number: int, cluster_heads: list[Node]) -> RoundRecord:
    living = alive_nodes(nodes)
    total = sum(node.energy for node in nodes)
    living_heads = sum(1 for head in cluster_heads if head.is_alive)
    return RoundRecord(round_number, len(living), total, living_heads)


def run_simulation(
    num_nodes: int,
    rounds: int,
    seed: int,
) -> tuple[list[Node], list[RoundRecord], dict[str, int | None], dict[int, list[Node]]]:
    rng = np.random.default_rng(seed)
    nodes = initialize_nodes(num_nodes, rng)
    records: list[RoundRecord] = []
    snapshots: dict[int, list[Node]] = {}
    snapshot_rounds = {1, max(1, rounds // 2), rounds}
    lifecycle = {
        "first_dead_round": None,
        "half_dead_round": None,
        "all_dead_round": None,
    }

    for round_index in range(rounds):
        round_number = round_index + 1
        if not alive_nodes(nodes):
            lifecycle["all_dead_round"] = lifecycle["all_dead_round"] or round_number - 1
            break

        reset_round_state(nodes)
        cluster_heads = elect_cluster_heads(nodes, round_index, P, rng)
        assign_clusters(nodes, cluster_heads)
        simulate_communication(nodes, cluster_heads, PACKET_SIZE, BASE_STATION)
        record = record_round(nodes, round_number, cluster_heads)
        records.append(record)

        dead_count = num_nodes - record.alive_nodes
        if dead_count >= 1 and lifecycle["first_dead_round"] is None:
            lifecycle["first_dead_round"] = round_number
        if dead_count >= math.ceil(num_nodes / 2) and lifecycle["half_dead_round"] is None:
            lifecycle["half_dead_round"] = round_number
        if record.alive_nodes == 0 and lifecycle["all_dead_round"] is None:
            lifecycle["all_dead_round"] = round_number

        if round_number in snapshot_rounds:
            snapshots[round_number] = clone_nodes(nodes)

        if record.alive_nodes == 0:
            break

    for snapshot_round in sorted(snapshot_rounds):
        if snapshot_round not in snapshots and records:
            snapshots[snapshot_round] = clone_nodes(nodes)

    return nodes, records, lifecycle, snapshots


def clone_nodes(nodes: list[Node]) -> list[Node]:
    return [Node(**node.__dict__) for node in nodes]


def prepare_pyplot(show: bool):
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_initial_distribution(
    nodes: list[Node],
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter([node.x for node in nodes], [node.y for node in nodes], c="#2f80ed", s=28, label="Sensor node")
    ax.scatter(*BASE_STATION, c="black", marker="*", s=180, label="Base station")
    style_network_axes(ax, "Initial WSN Node Distribution")
    ax.legend(loc="upper right")
    save_or_show(fig, output_dir / "initial_distribution.png", show)


def plot_cluster_snapshot(
    nodes: list[Node],
    round_number: int,
    output_dir: Path,
    show: bool,
) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 7))
    heads = [node for node in nodes if node.is_alive and node.is_CH]
    normal_nodes = [node for node in nodes if node.is_alive and not node.is_CH]
    dead_nodes = [node for node in nodes if not node.is_alive]
    head_by_id = {head.id: head for head in heads}

    for node in normal_nodes:
        head = head_by_id.get(node.cluster_id)
        if head is not None:
            ax.plot([node.x, head.x], [node.y, head.y], color="#9aa0a6", linestyle="--", linewidth=0.7, alpha=0.35)

    for head in heads:
        ax.plot([head.x, BASE_STATION[0]], [head.y, BASE_STATION[1]], color="#d62728", linewidth=0.9, alpha=0.55)

    if normal_nodes:
        ax.scatter([node.x for node in normal_nodes], [node.y for node in normal_nodes], c="#2f80ed", s=28, label="Normal node")
    if heads:
        ax.scatter([node.x for node in heads], [node.y for node in heads], c="#d62728", marker="^", s=70, label="Cluster head")
    if dead_nodes:
        ax.scatter([node.x for node in dead_nodes], [node.y for node in dead_nodes], c="#6b7280", marker="x", s=32, label="Dead node")
    ax.scatter(*BASE_STATION, c="black", marker="*", s=180, label="Base station")

    style_network_axes(ax, f"LEACH Clustering Result - Round {round_number}")
    ax.legend(loc="upper right")
    save_or_show(fig, output_dir / f"cluster_round_{round_number}.png", show)


def plot_alive_curve(records: list[RoundRecord], output_dir: Path, show: bool) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([record.round_number for record in records], [record.alive_nodes for record in records], color="#138a36", linewidth=2)
    ax.set_title("Alive Nodes Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Alive nodes")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, output_dir / "alive_nodes_curve.png", show)


def plot_energy_curve(records: list[RoundRecord], output_dir: Path, show: bool) -> None:
    plt = prepare_pyplot(show)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([record.round_number for record in records], [record.total_energy for record in records], color="#b45309", linewidth=2)
    ax.set_title("Total Remaining Energy Over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Total energy (J)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, output_dir / "total_energy_curve.png", show)


def style_network_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_xlim(-5, AREA_WIDTH + 5)
    ax.set_ylim(-5, max(AREA_HEIGHT, BASE_STATION[1]) + 10)
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


def write_summary_csv(records: list[RoundRecord], lifecycle: dict[str, int | None], output_dir: Path) -> None:
    with (output_dir / "simulation_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "round",
            "alive_nodes",
            "total_energy_j",
            "cluster_heads",
            "first_dead_round",
            "half_dead_round",
            "all_dead_round",
        ])
        for record in records:
            writer.writerow([
                record.round_number,
                record.alive_nodes,
                f"{record.total_energy:.8f}",
                record.cluster_heads,
                "" if lifecycle["first_dead_round"] is None else lifecycle["first_dead_round"],
                "" if lifecycle["half_dead_round"] is None else lifecycle["half_dead_round"],
                "" if lifecycle["all_dead_round"] is None else lifecycle["all_dead_round"],
            ])


def write_experiment_report(
    output_dir: Path,
    num_nodes: int,
    rounds: int,
    seed: int,
    records: list[RoundRecord],
    lifecycle: dict[str, int | None],
) -> None:
    final_record = records[-1] if records else RoundRecord(0, 0, 0.0, 0)
    report = f"""# WSN 节点分簇聚合模拟实验报告

## 1. 项目简介

本项目使用 Python 实现无线传感器网络 WSN 中经典的 LEACH 分簇聚合协议仿真。仿真在二维区域内随机部署传感器节点，通过周期性簇头选举、普通节点入簇、簇内数据传输、簇头数据聚合和簇头到基站传输，观察网络能量消耗和节点存活情况。

## 2. 实验参数

| 参数 | 取值 |
| --- | --- |
| 仿真区域 | {AREA_WIDTH:.0f} m x {AREA_HEIGHT:.0f} m |
| 节点数量 | {num_nodes} |
| 仿真轮数 | {rounds} |
| 随机种子 | {seed} |
| 初始能量 | {INIT_ENERGY} J |
| 期望簇头比例 | {P} |
| 数据包大小 | {PACKET_SIZE} bit |
| 基站位置 | {BASE_STATION} |
| 电路能耗 E_ELEC | {E_ELEC} J/bit |
| 放大器能耗 E_AMP | {E_AMP} J/bit/m^2 |
| 数据聚合能耗 E_DA | {E_DA} J/bit |

## 3. 算法流程

1. 随机生成传感器节点位置，并设置节点初始能量。
2. 每一轮根据 LEACH 阈值函数选举簇头节点。
3. 普通节点选择距离最近的簇头加入。
4. 普通节点向簇头发送数据，簇头接收并聚合数据。
5. 簇头将聚合后的数据发送到基站。
6. 更新节点剩余能量，统计存活节点数量和网络总能量。

## 4. 实验结果

程序运行后会在 `{output_dir}` 目录下生成以下文件：

- `initial_distribution.png`：初始节点分布图。
- `cluster_round_1.png`：第 1 轮分簇结果图。
- `cluster_round_{max(1, rounds // 2)}.png`：中间轮次分簇结果图。
- `cluster_round_{rounds}.png`：最后轮次分簇结果图。
- `alive_nodes_curve.png`：存活节点数量变化曲线。
- `total_energy_curve.png`：网络总剩余能量变化曲线。
- `simulation_summary.csv`：每轮统计数据。

本次仿真实际完成 {final_record.round_number} 轮，最终存活节点数为 {final_record.alive_nodes}，最终网络总剩余能量为 {final_record.total_energy:.6f} J。

| 生命周期指标 | 轮次 |
| --- | --- |
| 第一个节点死亡 | {format_lifecycle(lifecycle["first_dead_round"])} |
| 半数节点死亡 | {format_lifecycle(lifecycle["half_dead_round"])} |
| 全部节点死亡 | {format_lifecycle(lifecycle["all_dead_round"])} |

## 5. 总结

实验结果表明，LEACH 协议通过周期性簇头轮换，将长距离通信任务分摊到不同节点上，能够体现无线传感器网络中分簇路由的能量均衡思想。由于簇头需要承担接收、聚合和远距离发送任务，其单轮能耗通常高于普通节点，因此轮换簇头是延长网络生命周期的重要机制。
"""
    Path("Experiment_Report.md").write_text(report, encoding="utf-8")


def format_lifecycle(value: int | None) -> str:
    return "未发生" if value is None else str(value)


def validate_records(records: list[RoundRecord]) -> None:
    for previous, current in zip(records, records[1:]):
        if current.alive_nodes > previous.alive_nodes:
            raise RuntimeError("Alive node count increased unexpectedly.")
        if current.total_energy > previous.total_energy + 1e-12:
            raise RuntimeError("Total energy increased unexpectedly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEACH simulation for wireless sensor networks.")
    parser.add_argument("--nodes", type=int, default=100, help="Number of sensor nodes.")
    parser.add_argument("--rounds", type=int, default=100, help="Number of simulation rounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated charts and CSV summary.")
    parser.add_argument("--show", action="store_true", help="Display figures in addition to saving them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nodes <= 0:
        raise ValueError("--nodes must be greater than 0.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    rng = np.random.default_rng(args.seed)
    initial_nodes = initialize_nodes(args.nodes, rng)
    nodes, records, lifecycle, snapshots = run_simulation(args.nodes, args.rounds, args.seed)

    validate_records(records)
    plot_initial_distribution(initial_nodes, output_dir, args.show)
    for round_number in sorted(snapshots):
        plot_cluster_snapshot(snapshots[round_number], round_number, output_dir, args.show)
    plot_alive_curve(records, output_dir, args.show)
    plot_energy_curve(records, output_dir, args.show)
    write_summary_csv(records, lifecycle, output_dir)
    write_experiment_report(output_dir, args.nodes, args.rounds, args.seed, records, lifecycle)

    final_record = records[-1] if records else RoundRecord(0, 0, 0.0, 0)
    print("LEACH simulation completed.")
    print(f"Rounds simulated: {final_record.round_number}")
    print(f"Alive nodes: {final_record.alive_nodes}/{args.nodes}")
    print(f"Total remaining energy: {final_record.total_energy:.6f} J")
    print(f"Output directory: {output_dir}")
    print("Report: Experiment_Report.md")


if __name__ == "__main__":
    main()
