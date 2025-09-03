from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Union


# === Enumerations (mirror UI indices) ===

class SelectionMethod(Enum):
    ELITE = 0
    RANK = 1
    ROULETTE = 2


class MutationKind(Enum):
    BIT = 0
    BYTE = 1
    ALL_BYTES = 2


class FitnessFunctionType(Enum):
    TOPOLOGY_SAMPLE = 0
    TRIANGLE_MESH = 1
    QUADRIC_MESH = 2
    HEXAGON_MESH = 3
    CIRCLE = 4  # present in UI, not implemented in code-behind


class TranscriptionWay(Enum):
    RESETTABLE = 0
    CONTINUABLE = 1


class ViewMode(Enum):
    DISPLAY_NOTHING = 0
    DISPLAY_VERTICES = 1
    DISPLAY_EDGES = 2
    DISPLAY_EDGES_AND_VERTICES = 3
    DISPLAY_EDGES_VERTICES_WITH_STATE = 4


# === Parameter groups (dataclasses) ===

@dataclass
class GASettings:
    # XAML defaults or plausible from XAML
    selection_method: SelectionMethod = SelectionMethod.ELITE            # default: Elite (index 0)
    population_size: int = 100                                           # clamped [10..1000] in code-behind
    chromosome_start_length: int = 250                                   # derived max = 2 * start
    iteration_count: int = 100000                                        # 0 means unlimited
    random_selection_ratio: float = 0.0                                  # from XAML
    mutation_ratio: float = 0.05                                         # XAML empty; plausible GA default chosen
    crossing_ratio: float = 0.5                                          # from XAML
    active_gen_mutation_factor: float = 0.1                              # from XAML
    active_gen_mutation_kind: MutationKind = MutationKind.BYTE           # XAML SelectedIndex=1
    passive_gen_mutation_factor: float = 0.5                             # from XAML
    passive_gen_mutation_kind: MutationKind = MutationKind.ALL_BYTES     # XAML SelectedIndex=2
    max_degree_of_parallelism: int = -1                                  # -1 means default/unbounded
    experiments_repeats_count: int = 5                                   # from XAML

    @property
    def chromosome_max_length(self) -> int:
        """Derived: 2 * chromosome_start_length"""
        return 2 * self.chromosome_start_length


@dataclass
class FitnessParams:
    # None selected by default in XAML
    type: Optional[FitnessFunctionType] = None
    sample_index: Optional[int] = None  # 0..21 (mapped to specific sample graphs)

    shell_vertex_weight: float = 0.5                # tooltip: 0.0 exclude shell from fitness; 1.0 include fully
    faset3_penalty_probability: float = 1.0         # label shows "%"; UI default is 1
    guca_iteration_count: int = 150                 # from XAML
    max_vertex_count: int = 200                     # from XAML
    transcription_way: TranscriptionWay = TranscriptionWay.CONTINUABLE   # XAML SelectedIndex=1
    genome_length_penalty: bool = False             # from XAML
    not_planned_penalty: bool = True                # from XAML


@dataclass
class NumericalMethodParams:
    # Code constants + one value from UI
    one_iteration_time_step: float = 0.20           # code constant
    one_process_iteration_count: int = 20           # code constant
    outer_iteration_count: int = 20                 # from tbUnfoldingIterationCount (XAML default 20)


@dataclass
class ViewSettings:
    view_mode: ViewMode = ViewMode.DISPLAY_VERTICES  # XAML SelectedIndex=1
    initial_scale_after_draw: float = 3.0            # set in code after drawing


@dataclass
class LoggingSettings:
    file_name: str = "log.txt"  # code default


@dataclass
class AppConfig:
    ga: GASettings = field(default_factory=GASettings)
    fitness: FitnessParams = field(default_factory=FitnessParams)
    numerical: NumericalMethodParams = field(default_factory=NumericalMethodParams)
    view: ViewSettings = field(default_factory=ViewSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


# === Machine-readable parameter spec (name, type, possible values, default, notes) ===

PARAMETER_SPEC: Dict[str, List[Dict[str, Any]]] = {
    "ga": [
        {"name": "selection_method", "type": "SelectionMethod",
         "possible_values": [e.name for e in SelectionMethod],
         "default": GASettings().selection_method.name},

        {"name": "population_size", "type": "int",
         "possible_values": {"range": [10, 1000], "clamp": True},
         "default": GASettings().population_size},

        {"name": "chromosome_start_length", "type": "int",
         "possible_values": {"min": 0},
         "default": GASettings().chromosome_start_length},

        {"name": "chromosome_max_length", "type": "int",
         "possible_values": {"derived": "2 * chromosome_start_length"},
         "default": GASettings().chromosome_max_length},

        {"name": "iteration_count", "type": "int",
         "possible_values": {"min": 0, "note": "0 means unlimited"},
         "default": GASettings().iteration_count},

        {"name": "random_selection_ratio", "type": "float",
         "possible_values": {"recommended_range": [0.0, 1.0]},
         "default": GASettings().random_selection_ratio},

        {"name": "mutation_ratio", "type": "float",
         "possible_values": {"recommended_range": [0.0, 1.0]},
         "default": GASettings().mutation_ratio,
         "notes": "XAML had empty; using plausible default 0.05"},

        {"name": "crossing_ratio", "type": "float",
         "possible_values": {"recommended_range": [0.0, 1.0]},
         "default": GASettings().crossing_ratio},

        {"name": "active_gen_mutation_factor", "type": "float",
         "possible_values": {"recommended_range": [0.0, 1.0]},
         "default": GASettings().active_gen_mutation_factor},

        {"name": "active_gen_mutation_kind", "type": "MutationKind",
         "possible_values": [e.name for e in MutationKind],
         "default": GASettings().active_gen_mutation_kind.name},

        {"name": "passive_gen_mutation_factor", "type": "float",
         "possible_values": {"recommended_range": [0.0, 1.0]},
         "default": GASettings().passive_gen_mutation_factor},

        {"name": "passive_gen_mutation_kind", "type": "MutationKind",
         "possible_values": [e.name for e in MutationKind],
         "default": GASettings().passive_gen_mutation_kind.name},

        {"name": "max_degree_of_parallelism", "type": "int",
         "possible_values": {"note": "-1 means default/unbounded"},
         "default": GASettings().max_degree_of_parallelism},

        {"name": "experiments_repeats_count", "type": "int",
         "possible_values": {"min": 1},
         "default": GASettings().experiments_repeats_count},
    ],
    "fitness": [
        {"name": "type", "type": "Optional[FitnessFunctionType]",
         "possible_values": [e.name for e in FitnessFunctionType],
         "default": None,
         "notes": "CIRCLE present in UI but not implemented in C# switch-case"},

        {"name": "sample_index", "type": "Optional[int]",
         "possible_values": {"range": [0, 21], "note": "Maps to predefined graph samples"},
         "default": None},

        {"name": "shell_vertex_weight", "type": "float",
         "possible_values": {"range": [0.0, 1.0], "semantics": "0.0 exclude shell; 1.0 fully include"},
         "default": 0.5},

        {"name": "faset3_penalty_probability", "type": "float",
         "possible_values": {"range": [0.0, 100.0], "units": "%"},
         "default": 1.0},

        {"name": "guca_iteration_count", "type": "int",
         "possible_values": {"min": 1},
         "default": 150},

        {"name": "max_vertex_count", "type": "int",
         "possible_values": {"min": 1},
         "default": 200},

        {"name": "transcription_way", "type": "TranscriptionWay",
         "possible_values": [e.name for e in TranscriptionWay],
         "default": TranscriptionWay.CONTINUABLE.name},

        {"name": "genome_length_penalty", "type": "bool",
         "possible_values": [True, False],
         "default": False},

        {"name": "not_planned_penalty", "type": "bool",
         "possible_values": [True, False],
         "default": True},
    ],
    "numerical": [
        {"name": "one_iteration_time_step", "type": "float",
         "possible_values": {"fixed": True},
         "default": 0.20},

        {"name": "one_process_iteration_count", "type": "int",
         "possible_values": {"fixed": True},
         "default": 20},

        {"name": "outer_iteration_count", "type": "int",
         "possible_values": {"min": 1},
         "default": 20,
         "notes": "Taken from tbUnfoldingIterationCount (XAML=20). The helper Unfold() also uses 20."},
    ],
    "view": [
        {"name": "view_mode", "type": "ViewMode",
         "possible_values": [e.name for e in ViewMode],
         "default": ViewMode.DISPLAY_VERTICES.name},

        {"name": "initial_scale_after_draw", "type": "float",
         "possible_values": {"note": "Set by code after each draw"},
         "default": 3.0},
    ],
    "logging": [
        {"name": "file_name", "type": "str",
         "possible_values": {"any_path": True},
         "default": "log.txt"},
    ]
}


# Example: build the default config and turn it into a dict
DEFAULT_CONFIG = AppConfig()
DEFAULT_CONFIG_DICT: Dict[str, Any] = asdict(DEFAULT_CONFIG)

if __name__ == "__main__":
    # quick demo
    import json
    print(json.dumps(DEFAULT_CONFIG_DICT, indent=2, default=str))
    # You can also inspect PARAMETER_SPEC for UI/schema generation
