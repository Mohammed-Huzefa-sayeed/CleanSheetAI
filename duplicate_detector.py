import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from typing import Optional, List, Dict, Union, Tuple
from itertools import combinations
import hashlib
from collections import defaultdict

def _find_connected_components(nodes: list, edges: list) -> list[set]:
    """Find connected components in an undirected graph using standard library BFS."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = set()
    components = []
    
    for node in nodes:
        if node not in visited:
            component = set()
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                component.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components

class DuplicateGroup:
    """Represents a group of potential duplicate records."""
    def __init__(self, group_id: int, indices: list[int], similarity_scores: dict, match_type: str):
        self.group_id = group_id
        self.indices = indices  # row indices in the DataFrame
        self.similarity_scores = similarity_scores  # {column: score} for each compared column  
        self.match_type = match_type  # 'exact' or 'fuzzy'
        self.resolution = None  # 'keep_first', 'keep_last', 'merge', 'keep_all', None (unresolved)

class DuplicateDetector:
    """Comprehensive duplicate detection using exact and fuzzy matching."""
    
    def __init__(self, similarity_threshold: float = 85.0):
        self.similarity_threshold = similarity_threshold
        self.duplicate_groups: list[DuplicateGroup] = []
    
    def find_exact_duplicates(self, df: pd.DataFrame, columns: list[str] | None = None) -> list[DuplicateGroup]:
        """Find exact duplicate rows.
        If columns specified, only compare those columns.
        Returns list of DuplicateGroup objects with match_type='exact'.
        Each group contains all indices of duplicate rows (including the first occurrence)."""
        if df.empty or len(df) <= 1:
            return []
            
        compare_cols = columns if columns else df.columns.tolist()
        
        # We need to find all groups of duplicates
        # Group by the specified columns
        # Then filter groups with size > 1
        
        # Convert list/dict types to tuple to make them hashable if necessary, but standard df.groupby does okay with basic types.
        groups = []
        try:
            grouped = df.groupby(compare_cols)
        except Exception:
            # Fallback if unhashable types exist
            temp_df = df[compare_cols].astype(str)
            grouped = temp_df.groupby(compare_cols)
            
        group_id = 1
        for _, indices_in_group in grouped.groups.items():
            if len(indices_in_group) > 1:
                # Calculate basic similarity score (100.0 for exact matches)
                scores = {col: 100.0 for col in compare_cols}
                groups.append(DuplicateGroup(
                    group_id=group_id,
                    indices=list(indices_in_group),
                    similarity_scores=scores,
                    match_type='exact'
                ))
                group_id += 1
                
        self.duplicate_groups.extend(groups)
        return groups
    
    def _calculate_similarity(self, row1: pd.Series, row2: pd.Series, columns: list[str]) -> tuple[float, dict]:
        """Helper to calculate similarity between two rows across specific columns."""
        col_scores = {}
        total_score = 0.0
        valid_cols = 0
        
        for col in columns:
            v1 = row1[col]
            v2 = row2[col]
            
            # Handle NaNs
            if pd.isna(v1) and pd.isna(v2):
                score = 100.0
            elif pd.isna(v1) or pd.isna(v2):
                score = 0.0
            else:
                v1_str = str(v1).lower().strip()
                v2_str = str(v2).lower().strip()
                score = fuzz.token_sort_ratio(v1_str, v2_str)
                
            col_scores[col] = score
            total_score += score
            valid_cols += 1
            
        avg_score = total_score / valid_cols if valid_cols > 0 else 0.0
        return avg_score, col_scores
    
    def find_fuzzy_duplicates(self, df: pd.DataFrame, columns: list[str], 
                               threshold: float | None = None,
                               max_comparisons: int = 50000) -> list[DuplicateGroup]:
        """Find fuzzy duplicate rows using RapidFuzz."""
        if df.empty or len(df) <= 1:
            return []
            
        thresh = threshold if threshold is not None else self.similarity_threshold
        
        # Limit the rows to avoid too many comparisons if necessary
        # combinations(n, 2) = n(n-1)/2
        n_rows = len(df)
        total_possible = n_rows * (n_rows - 1) // 2
        
        indices_to_compare = list(df.index)
        if total_possible > max_comparisons:
            # Take a sample or just top N to fit max_comparisons
            import math
            # n^2/2 = max_comp -> n = sqrt(2*max_comp)
            limit = int(math.sqrt(2 * max_comparisons))
            indices_to_compare = indices_to_compare[:limit]
        
        # Graph approach for connected components without networkx
        edges = []
        edge_scores = {}
        
        for idx1, idx2 in combinations(indices_to_compare, 2):
            row1 = df.loc[idx1]
            row2 = df.loc[idx2]
            
            avg_score, col_scores = self._calculate_similarity(row1, row2, columns)
            if avg_score >= thresh:
                edges.append((idx1, idx2))
                edge_scores[(idx1, idx2)] = col_scores
                
        # Find components
        groups = []
        group_id = 1
        for component in _find_connected_components(indices_to_compare, edges):
            if len(component) > 1:
                # Create a generalized score for the group
                # Here we just take the first edge's score as representative
                nodes = list(component)
                # Find an edge in this component
                rep_scores = {}
                for u, v in edge_scores:
                    if u in nodes and v in nodes:
                        rep_scores = edge_scores[(u, v)]
                        break
                        
                groups.append(DuplicateGroup(
                    group_id=group_id,
                    indices=nodes,
                    similarity_scores=rep_scores,
                    match_type='fuzzy'
                ))
                group_id += 1
                
        self.duplicate_groups.extend(groups)
        return groups
    
    def get_duplicate_summary(self, df: pd.DataFrame, groups: list[DuplicateGroup]) -> dict:
        """Generate summary of duplicate detection results."""
        exact_groups = [g for g in groups if g.match_type == 'exact']
        fuzzy_groups = [g for g in groups if g.match_type == 'fuzzy']
        
        unique_dup_indices = set()
        for g in groups:
            unique_dup_indices.update(g.indices)
            
        columns_involved = set()
        highest_sim = 0.0
        lowest_sim = 100.0
        
        for g in groups:
            if g.similarity_scores:
                columns_involved.update(g.similarity_scores.keys())
                avg_score = sum(g.similarity_scores.values()) / len(g.similarity_scores)
                highest_sim = max(highest_sim, avg_score)
                lowest_sim = min(lowest_sim, avg_score)
                
        return {
            'total_groups': len(groups),
            'total_duplicate_rows': len(unique_dup_indices),
            'exact_groups': len(exact_groups),
            'fuzzy_groups': len(fuzzy_groups),
            'columns_involved': list(columns_involved),
            'highest_similarity': highest_sim if groups else None,
            'lowest_similarity': lowest_sim if groups else None
        }
    
    def get_comparison_view(self, df: pd.DataFrame, group: DuplicateGroup) -> pd.DataFrame:
        """Get a side-by-side view of records in a duplicate group.
        Handles both label-based indices and positional indices safely,
        even on filtered or non-consecutive DataFrames.
        """
        # First check which indices exist in df.index (label-based)
        valid_indices = [idx for idx in group.indices if idx in df.index]
        if valid_indices:
            comp_df = df.loc[valid_indices].copy()
        else:
            # Fallback to positional iloc if group.indices were integer positions within bounds
            valid_pos = [idx for idx in group.indices if isinstance(idx, (int, np.integer)) and 0 <= idx < len(df)]
            if valid_pos:
                comp_df = df.iloc[valid_pos].copy()
            else:
                comp_df = pd.DataFrame(columns=df.columns)

        comp_df['_row_index'] = comp_df.index
        # Use average similarity from the group if available
        avg_sim = sum(group.similarity_scores.values()) / len(group.similarity_scores) if group.similarity_scores else 100.0
        comp_df['_similarity'] = avg_sim
        return comp_df
    
    def resolve_group(self, df: pd.DataFrame, group: DuplicateGroup, 
                       resolution: str, keep_index: int | None = None,
                       merge_rules: dict | None = None) -> pd.DataFrame:
        """Resolve a duplicate group."""
        if resolution == 'keep_all':
            group.resolution = 'keep_all'
            return df.copy()
            
        new_df = df.copy()
        indices_to_drop = []
        indices = group.indices
        valid_indices = [i for i in indices if i in new_df.index]
        if not valid_indices:
            return new_df


        if resolution == 'keep_first':
            indices_to_drop = valid_indices[1:]
        elif resolution == 'keep_last':
            indices_to_drop = valid_indices[:-1]
        elif resolution == 'keep_specific':
            if keep_index is not None and keep_index in valid_indices:
                indices_to_drop = [i for i in valid_indices if i != keep_index]
            else:
                raise ValueError("keep_index must be specified and must be within group.indices")
        elif resolution == 'merge':
            if not merge_rules:
                raise ValueError("merge_rules must be provided for merge resolution")
                
            base_idx = valid_indices[0]
            indices_to_drop = valid_indices[1:]
            
            for col, rule in merge_rules.items():
                if col not in new_df.columns:
                    continue
                
                values = [new_df.loc[i, col] for i in valid_indices if not pd.isna(new_df.loc[i, col])]
                if not values:
                    continue

                    
                if rule == 'first':
                    merged_val = values[0]
                elif rule == 'last':
                    merged_val = values[-1]
                elif rule == 'longest':
                    merged_val = max(values, key=lambda x: len(str(x)))
                elif rule == 'shortest':
                    merged_val = min(values, key=lambda x: len(str(x)))
                elif rule == 'concatenate':
                    merged_val = " | ".join(map(str, values))
                else:
                    merged_val = values[0]
                    
                new_df.at[base_idx, col] = merged_val
        else:
            raise ValueError(f"Unknown resolution type: {resolution}")
            
        valid_drops = [idx for idx in indices_to_drop if idx in new_df.index]
        if valid_drops:
            new_df = new_df.drop(index=valid_drops)
        group.resolution = resolution
        return new_df

    
    def remove_exact_duplicates(self, df: pd.DataFrame, columns: list[str] | None = None,
                                 keep: str = 'first') -> tuple[pd.DataFrame, int]:
        """Simple exact duplicate removal."""
        initial_len = len(df)
        subset = columns if columns else None
        
        if keep == 'first' or keep == 'last' or keep is False:
            new_df = df.drop_duplicates(subset=subset, keep=keep)
        else:
            new_df = df.drop_duplicates(subset=subset, keep='first')
            
        count_removed = initial_len - len(new_df)
        return new_df.copy(), count_removed
    
    def get_field_match_details(self, df: pd.DataFrame, idx1: int, idx2: int, 
                                 columns: list[str]) -> list[dict]:
        """Compare two specific records field by field."""
        if idx1 in df.index:
            row1 = df.loc[idx1]
        elif isinstance(idx1, (int, np.integer)) and 0 <= idx1 < len(df):
            row1 = df.iloc[idx1]
        else:
            raise KeyError(f"Index {idx1} not found in DataFrame index or positions")

        if idx2 in df.index:
            row2 = df.loc[idx2]
        elif isinstance(idx2, (int, np.integer)) and 0 <= idx2 < len(df):
            row2 = df.iloc[idx2]
        else:
            raise KeyError(f"Index {idx2} not found in DataFrame index or positions")
        
        details = []
        for col in columns:
            v1 = row1[col]
            v2 = row2[col]
            
            if pd.isna(v1) and pd.isna(v2):
                match_type = 'exact'
                score = 100.0
            elif pd.isna(v1) or pd.isna(v2):
                match_type = 'mismatch'
                score = 0.0
            else:
                v1_str = str(v1)
                v2_str = str(v2)
                
                if v1 == v2 or v1_str == v2_str:
                    match_type = 'exact'
                    score = 100.0
                else:
                    score = fuzz.token_sort_ratio(v1_str.lower(), v2_str.lower())
                    match_type = 'fuzzy' if score >= self.similarity_threshold else 'mismatch'
                    
            details.append({
                'column': col,
                'value1': v1,
                'value2': v2,
                'match_type': match_type,
                'similarity_score': score
            })
            
        return details
    
    def _soundex(self, s: str) -> str:
        """Simple soundex implementation."""
        if not s:
            return "0000"
        s = str(s).upper()
        s = "".join([c for c in s if c.isalpha()])
        if not s:
            return "0000"
            
        first_letter = s[0]
        mapping = {
            "B": "1", "F": "1", "P": "1", "V": "1",
            "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
            "D": "3", "T": "3",
            "L": "4",
            "M": "5", "N": "5",
            "R": "6"
        }
        
        encoded = first_letter
        for char in s[1:]:
            if char in mapping:
                code = mapping[char]
                if code != encoded[-1]:
                    encoded += code
                    
        encoded = encoded.replace(mapping.get(first_letter, ""), "", 1) if first_letter in mapping and len(encoded) > 1 and encoded[1] == mapping[first_letter] else encoded
        encoded = encoded.replace("0", "")
        
        # Ensure length 4
        return (encoded + "0000")[:4]

    def create_blocking_key(self, value: str, method: str = 'first3') -> str:
        """Create a blocking key for efficient comparison."""
        if pd.isna(value):
            return "NULL"
            
        val_str = str(value).strip().lower()
        if not val_str:
            return "EMPTY"
            
        if method == 'first3':
            return val_str[:3]
        elif method == 'soundex':
            return self._soundex(val_str)
        elif method == 'ngram':
            # 3-gram based blocking key
            if len(val_str) < 3:
                return val_str
            return val_str[:3]
        else:
            # Fallback
            return val_str[:3]
            
    def find_fuzzy_duplicates_blocked(self, df: pd.DataFrame, columns: list[str],
                                       blocking_column: str,
                                       blocking_method: str = 'first3',
                                       threshold: float | None = None) -> list[DuplicateGroup]:
        """Find fuzzy duplicates using blocking to improve performance."""
        if df.empty or len(df) <= 1:
            return []
            
        thresh = threshold if threshold is not None else self.similarity_threshold
        
        # Apply blocking
        blocks = {}
        for idx in df.index:
            val = df.loc[idx, blocking_column]
            key = self.create_blocking_key(val, method=blocking_method)
            if key not in blocks:
                blocks[key] = []
            blocks[key].append(idx)
            
        groups = []
        group_id = 1
        
        for key, indices in blocks.items():
            if len(indices) <= 1:
                continue
                
            # Perform fuzzy matching within the block without networkx
            edges = []
            edge_scores = {}
            
            for idx1, idx2 in combinations(indices, 2):
                row1 = df.loc[idx1]
                row2 = df.loc[idx2]
                
                avg_score, col_scores = self._calculate_similarity(row1, row2, columns)
                if avg_score >= thresh:
                    edges.append((idx1, idx2))
                    edge_scores[(idx1, idx2)] = col_scores
                    
            for component in _find_connected_components(indices, edges):
                if len(component) > 1:
                    nodes = list(component)
                    rep_scores = {}
                    for u, v in edge_scores:
                        if u in nodes and v in nodes:
                            rep_scores = edge_scores[(u, v)]
                            break
                            
                    groups.append(DuplicateGroup(
                        group_id=group_id,
                        indices=nodes,
                        similarity_scores=rep_scores,
                        match_type='fuzzy'
                    ))
                    group_id += 1
                    
        self.duplicate_groups.extend(groups)
        return groups
