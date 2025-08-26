#
# A Python adaptation of the GUCA (Graph Unfolding Cellular Automata) engine,
# referencing concepts from gum.ts. Uses networkx internally via GUMGraph.
#
import enum
from typing import List, Optional
from gum_graph import GUMGraph

# --------------------------------------------------------------------
# 1) NodeState
#    In TypeScript gum.ts, NodeState is an enum with many possible values
#    (A, B, C, D...) plus -1 for Unknown, 0 for Ignored, etc.
#    Here, we illustrate just a few. Feel free to expand/modify as needed.
# --------------------------------------------------------------------
class NodeState(enum.IntEnum):
    UNKNOWN = -1
    IGNORED = 0
    A = 1
    B = 2
    C = 3
    # ... add more states as needed, or dynamically handle them as ints


# --------------------------------------------------------------------
# 2) OperationKind
#    This parallels "OperationKindEnum" or "OperationKind" from gum.ts
# --------------------------------------------------------------------
class OperationKind(enum.IntEnum):
    TURN_TO_STATE = 0
    TRY_TO_CONNECT_WITH = 1
    TRY_TO_CONNECT_WITH_NEAREST = 2
    GIVE_BIRTH = 3
    GIVE_BIRTH_CONNECTED = 4
    DIE = 5
    DISCONNECT_FROM = 6
    # Feel free to add more if your gum.ts had additional cases


# --------------------------------------------------------------------
# 3) TranscriptionWay
#    gum.ts references the concept of reading the "rule table" from the top each time,
#    or continuing from the next rule after a match.
# --------------------------------------------------------------------
class TranscriptionWay(enum.IntEnum):
    RESETTABLE = 0
    CONTINUABLE = 1


# --------------------------------------------------------------------
# 4) OperationCondition
#    This matches the "condition" part of a Rule: what currentState, priorState,
#    connection counts, etc. must a node have for the rule to apply.
#    In gum.ts, we see fields for AllConnectionsCount_GE, etc.
# --------------------------------------------------------------------
class OperationCondition:
    def __init__(self,
                 current_state: NodeState = NodeState.IGNORED,
                 prior_state: NodeState = NodeState.UNKNOWN,
                 connections_ge: int = -1,  # -1 => ignore
                 connections_le: int = -1,  # -1 => ignore
                 parents_ge: int = -1,      # -1 => ignore
                 parents_le: int = -1       # -1 => ignore
                 ):
        self.current_state = current_state
        self.prior_state = prior_state
        self.connections_ge = connections_ge
        self.connections_le = connections_le
        self.parents_ge = parents_ge
        self.parents_le = parents_le

    def __repr__(self):
        return (f"OperationCondition("
                f"cur={self.current_state}, prior={self.prior_state}, "
                f"con_ge={self.connections_ge}, con_le={self.connections_le}, "
                f"par_ge={self.parents_ge}, par_le={self.parents_le})")


# --------------------------------------------------------------------
# 5) Operation
#    Contains the kind of operation (e.g. TURN_TO_STATE) and an operand (target state).
# --------------------------------------------------------------------
class Operation:
    def __init__(self,
                 kind: OperationKind = OperationKind.TURN_TO_STATE,
                 operand_state: NodeState = NodeState.IGNORED):
        self.kind = kind
        self.operand_state = operand_state

    def __repr__(self):
        return f"Operation(kind={self.kind}, operand={self.operand_state})"


# --------------------------------------------------------------------
# 6) Rule
#    Ties a Condition to an Operation.
#    gum.ts calls it "ChangeTableItem"; we can call it "Rule".
# --------------------------------------------------------------------
class Rule:
    def __init__(self,
                 condition: OperationCondition,
                 operation: Operation):
        self.condition = condition
        self.operation = operation
        self.is_enabled = True
        # 'was_active' can track if the rule was triggered at least once in a run
        self.was_active = False

    def __repr__(self):
        return f"Rule({self.condition} => {self.operation}, enabled={self.is_enabled})"


# --------------------------------------------------------------------
# 7) RuleTable
#    A collection (list) of rules, possibly hashed by currentState for faster lookup.
#    For simplicity, we store them in a Python list. In gum.ts, it was "ChangeTable".
# --------------------------------------------------------------------
class RuleTable:
    def __init__(self):
        self.rules: List[Rule] = []
        self.is_prepared = False  # If you want to do indexing or hashing

    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        self.is_prepared = False

    def prepare_for_search(self):
        # If needed, build a hash table or sort by current_state
        # For large rule sets, you can speed up rule lookup.
        self.is_prepared = True

    def reset_rule_usage(self):
        # Clear was_active flags, etc.
        for r in self.rules:
            r.was_active = False

    def __iter__(self):
        return iter(self.rules)

    def __len__(self):
        return len(self.rules)


# --------------------------------------------------------------------
# 8) GUCAEngine
#    The main "Graph Unfolding Machine". We store:
#      - A GUMGraph instance
#      - A RuleTable
#      - Settings for maxSteps, transcriptionWay, etc.
#    Then we implement run() or run_one_step() to evolve the graph.
# --------------------------------------------------------------------
class GUCAEngine:
    def __init__(self,
                 gum_graph: GUMGraph,
                 rule_table: RuleTable,
                 max_steps: int = 20,
                 transcription_way: TranscriptionWay = TranscriptionWay.RESETTABLE):
        self.gum_graph = gum_graph
        self.rule_table = rule_table
        self.max_steps = max_steps
        self.transcription_way = transcription_way

        self._passed_steps = 0
        self._without_any_operation_counter = 0
        self._last_applied_rule_index = dict()  # track where we left off per-node if CONTINUABLE

    @property
    def passed_steps(self):
        return self._passed_steps

    def reset(self):
        """
        Reset counters and statuses in preparation for a new run.
        """
        self._passed_steps = 0
        self._without_any_operation_counter = 0
        self._last_applied_rule_index.clear()
        self.rule_table.reset_rule_usage()
        # Optionally reset prior states or anything else in the graph

    def run(self):
        """
        Run the GUCA evolution for up to self.max_steps or until no operation occurs.
        """
        self.reset()  # in case you want a fresh start
        while self._passed_steps < self.max_steps:
            changes_occurred = self.run_one_step()
            self._passed_steps += 1

            if not changes_occurred:
                self._without_any_operation_counter += 1
                # If no changes occur for multiple consecutive steps, we can break early
                if self._without_any_operation_counter >= 2:
                    break
            else:
                self._without_any_operation_counter = 0

        # Optionally remove nodes marked as deleted:
        self._remove_deleted_nodes()

    def run_one_step(self) -> bool:
        """
        Execute one iteration:
          1. Save each node's current state as 'priorState'
          2. For each node (that is not marked deleted), apply first matching rule
          3. Perform the requested operation (turn, birth, etc.)

        Returns True if at least one operation is applied.
        """
        # Step 1: Save states
        for node_id in list(self.gum_graph.get_underlying_graph().nodes):
            if self.gum_graph.is_deleted(node_id):
                continue
            # set node's priorState to the current state
            current = self.gum_graph.get_state(node_id)
            self.gum_graph.set_prior_state(node_id, current)

        # Step 2: For each node, find the first matching rule and apply
        any_change = False
        for node_id in list(self.gum_graph.get_underlying_graph().nodes):
            if self.gum_graph.is_deleted(node_id):
                continue

            rule_index, matching_rule = self._find_applicable_rule(node_id)
            if matching_rule:
                # Mark that we used that rule
                matching_rule.was_active = True
                # Apply the operation
                if self._apply_operation(node_id, matching_rule.operation):
                    any_change = True

                # If CONTINUABLE, store the next index as rule_index+1
                if self.transcription_way == TranscriptionWay.CONTINUABLE:
                    self._last_applied_rule_index[node_id] = rule_index + 1
            else:
                # no rule found
                if self.transcription_way == TranscriptionWay.CONTINUABLE:
                    # wrap around to 0 for next iteration
                    self._last_applied_rule_index[node_id] = 0

        # Step 3: Remove physically any nodes that are "markedAsDeleted"
        #   (some GUCA versions delay removal until after the entire iteration).
        return any_change

    def _find_applicable_rule(self, node_id) -> (int, Optional[Rule]):
        """
        Search through the RuleTable for the first matching rule.
        If self.transcription_way == CONTINUABLE, we resume from
        the last triggered rule index for that node.
        """
        current_state = self.gum_graph.get_state(node_id)
        prior_state = self.gum_graph.get_prior_state(node_id)
        deg = self.gum_graph.node_degree(node_id)
        parents = self.gum_graph.node_parents_count(node_id)

        start_index = 0
        if (self.transcription_way == TranscriptionWay.CONTINUABLE
                and node_id in self._last_applied_rule_index):
            start_index = self._last_applied_rule_index[node_id]

        # We'll do a single pass from start_index to end, then 0 to start_index-1 if needed
        rule_range = list(range(len(self.rule_table)))
        # rotate so we start at 'start_index'
        rule_range = rule_range[start_index:] + rule_range[:start_index]

        for i in rule_range:
            rule = self.rule_table.rules[i]
            if not rule.is_enabled:
                continue

            cond = rule.condition
            # Check condition
            if cond.current_state != NodeState.IGNORED:
                if cond.current_state != current_state:
                    continue

            if cond.prior_state != NodeState.IGNORED:
                if cond.prior_state != prior_state:
                    continue

            if cond.connections_ge != -1:
                if deg < cond.connections_ge:
                    continue

            if cond.connections_le != -1:
                if deg > cond.connections_le:
                    continue

            if cond.parents_ge != -1:
                if parents < cond.parents_ge:
                    continue

            if cond.parents_le != -1:
                if parents > cond.parents_le:
                    continue

            # If all checks pass, this rule applies
            # figure out the actual index in the table
            actual_index = (start_index + i) % len(self.rule_table)
            return (actual_index, rule)

        return (-1, None)

    def _apply_operation(self, node_id, operation: Operation) -> bool:
        """
        Execute the operation on the given node. This could lead to changes in the graph.
        Return True if something changed; False otherwise.
        """
        op_kind = operation.kind
        operand = operation.operand_state
        changed = False

        if op_kind == OperationKind.TURN_TO_STATE:
            self.gum_graph.set_state(node_id, operand)
            changed = True

        elif op_kind == OperationKind.TRY_TO_CONNECT_WITH:
            # Connect this node to all existing nodes with saved_state == operand
            # or any logic you prefer, e.g. only one node.
            for other_id in self.gum_graph.get_underlying_graph().nodes:
                if other_id == node_id:
                    continue
                if self.gum_graph.is_deleted(other_id):
                    continue
                if self.gum_graph.get_state(other_id) == operand:
                    deg1 = self.gum_graph.node_degree(node_id)
                    deg2 = self.gum_graph.node_degree(other_id)
                    # check max connections in add_edge
                    self.gum_graph.add_edge(node_id, other_id)
                    if self.gum_graph.node_degree(node_id) != deg1 or \
                       self.gum_graph.node_degree(other_id) != deg2:
                        changed = True

        elif op_kind == OperationKind.GIVE_BIRTH:
            # If there's room, create a new node with state=operand
            # The new node's parentsCount = parent's parentsCount+1
            parent_par = self.gum_graph.node_parents_count(node_id)
            new_id = f"child_{node_id}_{self._passed_steps}_{len(self.gum_graph)}"
            node_added = self.gum_graph.add_node(new_id, state=operand,
                                                 prior_state=NodeState.UNKNOWN,
                                                 parents_count=parent_par + 1)
            if node_added:
                changed = True

        elif op_kind == OperationKind.GIVE_BIRTH_CONNECTED:
            # Like GIVE_BIRTH, but also connect the new node to node_id
            parent_par = self.gum_graph.node_parents_count(node_id)
            new_id = f"child_{node_id}_{self._passed_steps}_{len(self.gum_graph)}"
            node_added = self.gum_graph.add_node(new_id, state=operand,
                                                 prior_state=NodeState.UNKNOWN,
                                                 parents_count=parent_par + 1)
            if node_added:
                # Connect
                self.gum_graph.add_edge(node_id, new_id)
                changed = True

        elif op_kind == OperationKind.DIE:
            self.gum_graph.mark_as_deleted(node_id)
            changed = True

        elif op_kind == OperationKind.DISCONNECT_FROM:
            # Remove edges with nodes in state=operand
            neighbors = self.gum_graph.get_neighbors(node_id)
            for other_id in neighbors:
                if self.gum_graph.get_state(other_id) == operand:
                    self.gum_graph.remove_edge(node_id, other_id)
                    changed = True

        elif op_kind == OperationKind.TRY_TO_CONNECT_WITH_NEAREST:
            # This might require BFS or distance checks. For simplicity, here's a naive approach:
            # "Nearest" could be the first node found with the operand state.
            # Real logic might require BFS, geometry checks, etc.
            for other_id in self.gum_graph.get_underlying_graph().nodes:
                if other_id == node_id:
                    continue
                if self.gum_graph.get_state(other_id) == operand:
                    # Connect just the first match
                    before_deg = self.gum_graph.node_degree(node_id)
                    self.gum_graph.add_edge(node_id, other_id)
                    after_deg = self.gum_graph.node_degree(node_id)
                    if after_deg > before_deg:
                        changed = True
                    break

        else:
            pass  # Not recognized or not implemented

        return changed

    def _remove_deleted_nodes(self):
        """
        Physically remove nodes from the graph that are 'markedAsDeleted'.
        """
        to_remove = []
        for node_id in list(self.gum_graph.get_underlying_graph().nodes):
            if self.gum_graph.is_deleted(node_id):
                to_remove.append(node_id)
        for n in to_remove:
            self.gum_graph.remove_node(n)
