"""构建高质量编程推理蒸馏数据集
用最新 P2 GGUF 模型生成, 严格过滤, 输出 instruction/reasoning/response 格式.

用法:
  python build_coding_dataset.py                    # 用 P2 模型
  python build_coding_dataset.py --model path.gguf  # 指定模型
  python build_coding_dataset.py --resume           # 断点续传
"""
import argparse, json, os, re, sys, time, ast, subprocess, tempfile

os.environ["PATH"] = (
    "F:/Python/Lib/site-packages/llama_cpp/lib"
    + os.pathsep + os.environ.get("PATH", ""))

from llama_cpp import Llama

# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL = "./output/p2-q4_k_m.gguf"
OUTPUT_FILE = "data/coding_reasoning_dataset.jsonl"
CHECKPOINT_FILE = "data/coding_reasoning_ckpt.json"
MAX_TOKENS = 1024
TEMPERATURE = 0.1

# ============================================================
# 禁止调用的库
# ============================================================
BANNED_IMPORTS = [
    "aioredis", "redis", "redislite",
    "numpy", "np", "pandas", "pd", "sklearn", "scipy",
    "tensorflow", "torch", "keras",
    "flask", "django", "fastapi",
    "requests", "httpx", "aiohttp",
    "sqlalchemy", "pymongo", "psycopg2",
    "celery", "rabbitmq",
    "eventemitter3", "rxjs", "mitt",
    "beautifulsoup4", "bs4", "scrapy",
]

# 只允许的标准库
ALLOWED_IMPORTS = {
    "collections", "heapq", "bisect", "itertools", "functools",
    "typing", "math", "random", "string", "re",  # re 允许用于字符串题
    "json", "copy", "operator", "abc", "enum",
    "threading", "queue", "time", "hashlib",
    "asyncio", "dataclasses", "contextlib",
}

# ============================================================
# 300 道编程题 (从 cot_distill_v2 提取)
# ============================================================
PROBLEMS = [
    # === String (40) ===
    ("e_str_01", "easy", "string", "Write function reverse_string(s) returning string reversed without [::-1] or reversed()."),
    ("e_str_02", "easy", "string", "Write function is_anagram(s, t) checking if t is anagram of s in O(n)."),
    ("e_str_03", "easy", "string", "Write function first_unique_char(s) returning index of first non-repeating character, or -1."),
    ("e_str_04", "easy", "string", "Write function longest_common_prefix(strs) returning longest common prefix among array of strings."),
    ("e_str_05", "easy", "string", "Write function is_valid(s) checking if string of parentheses ()[]{} is valid."),
    ("e_str_06", "easy", "string", "Write function count_vowels(s) returning count of vowels case-insensitive."),
    ("e_str_07", "easy", "string", "Write function reverse_vowels(s) reversing only vowels in string."),
    ("e_str_08", "easy", "string", "Write function compress(s) compressing string by counting consecutive chars, e.g. aabbb -> a2b3."),
    ("e_str_09", "easy", "string", "Write function is_palindrome_str(s) checking if string is palindrome, ignoring non-alphanumeric."),
    ("e_str_10", "easy", "string", "Write function str_str(haystack, needle) returning index of first occurrence of needle, or -1."),
    ("e_str_11", "easy", "string", "Write function length_of_last_word(s) returning length of last word in string."),
    ("e_str_12", "easy", "string", "Write function to_lower_case(s) converting string to lowercase without built-in lower()."),
    ("e_str_13", "easy", "string", "Write function is_subsequence(s, t) checking if s is subsequence of t."),
    ("e_str_14", "easy", "string", "Write function most_common_char(s) returning the most frequent character in string."),
    ("e_str_15", "easy", "string", "Write function capitalize_words(s) capitalizing first letter of each word."),
    ("e_str_16", "easy", "string", "Write function remove_spaces(s) removing all whitespace from string."),
    ("e_str_17", "easy", "string", "Write function count_words(s) counting number of words in string."),
    ("e_str_18", "easy", "string", "Write function is_rotation(s1, s2) checking if s2 is rotation of s1."),
    ("e_str_19", "easy", "string", "Write function replace_spaces(s) replacing all spaces with %20 in-place."),
    ("e_str_20", "easy", "string", "Write function unique_chars(s) returning True if all characters in string are unique."),
    ("m_str_01", "medium", "string", "Write function length_of_longest_substring(s) returning longest substring without repeating chars."),
    ("m_str_02", "medium", "string", "Write function group_anagrams(strs) grouping anagrams together."),
    ("m_str_03", "medium", "string", "Write function longest_palindrome(s) returning longest palindromic substring, expand around center."),
    ("m_str_04", "medium", "string", "Write function my_atoi(s) converting string to integer handling whitespace, sign, overflow."),
    ("m_str_05", "medium", "string", "Write function int_to_roman(num) converting integer 1-3999 to Roman numeral."),
    ("m_str_06", "medium", "string", "Write function decode_string(s) decoding 3[a]2[bc] -> aaabcbc using stack."),
    ("m_str_07", "medium", "string", "Write function simplify_path(path) simplifying Unix absolute path."),
    ("m_str_08", "medium", "string", "Write function generate_parentheses(n) generating all valid combinations of n pairs of parentheses."),
    ("m_str_09", "medium", "string", "Write function letter_combinations(digits) returning all letter combinations of phone number."),
    ("m_str_10", "medium", "string", "Write function word_break(s, wordDict) checking if s can be segmented into dictionary words."),
    ("m_str_11", "medium", "string", "Write function min_window(s, t) finding minimum window in s containing all characters of t."),
    ("m_str_12", "medium", "string", "Write function find_all_anagrams(s, p) finding all start indices of p's anagrams in s."),
    ("m_str_13", "medium", "string", "Write function longest_repeating_replacement(s, k) finding longest substring with at most k replacements."),
    ("m_str_14", "medium", "string", "Write function multiply_strings(num1, num2) multiplying two non-negative integer strings."),
    ("m_str_15", "medium", "string", "Write function count_and_say(n) returning n-th term of count-and-say sequence."),
    ("m_str_16", "medium", "string", "Write function full_justify(words, maxWidth) formatting text like a text editor."),
    ("m_str_17", "medium", "string", "Write function compare_versions(version1, version2) comparing two version strings."),
    ("m_str_18", "medium", "string", "Write function restore_ip_addresses(s) returning all valid IP addresses from string."),
    ("m_str_19", "medium", "string", "Write function longest_valid_parentheses(s) finding length of longest valid parentheses substring."),
    ("m_str_20", "medium", "string", "Write function zigzag_convert(s, numRows) converting string to zigzag pattern."),
    # === Array (50) ===
    ("e_arr_01", "easy", "array", "Write function max_subarray(nums) returning max contiguous subarray sum using Kadane's algorithm."),
    ("e_arr_02", "easy", "array", "Write function contains_duplicate(nums) returning True if any value appears twice."),
    ("e_arr_03", "easy", "array", "Write function remove_duplicates(nums) removing duplicates from sorted array in-place."),
    ("e_arr_04", "easy", "array", "Write function move_zeroes(nums) moving all zeros to end preserving order, in-place."),
    ("e_arr_05", "easy", "array", "Write function plus_one(digits) incrementing large integer represented as digit array."),
    ("e_arr_06", "easy", "array", "Write function majority_element(nums) returning element appearing more than n/2 times."),
    ("e_arr_07", "easy", "array", "Write function missing_number(nums) finding missing number in array containing 0..n."),
    ("e_arr_08", "easy", "array", "Write function two_sum(nums, target) returning indices of two numbers that add up to target."),
    ("e_arr_09", "easy", "array", "Write function max_profit(prices) finding max profit from one buy-sell of stock."),
    ("e_arr_10", "easy", "array", "Write function intersect(nums1, nums2) returning intersection of two arrays."),
    ("e_arr_11", "easy", "array", "Write function single_number(nums) finding element appearing once when all others appear twice."),
    ("e_arr_12", "easy", "array", "Write function merge_sorted(nums1, m, nums2, n) merging nums2 into nums1 in-place."),
    ("e_arr_13", "easy", "array", "Write function pascal_row(rowIndex) returning rowIndex-th row of Pascal's triangle."),
    ("e_arr_14", "easy", "array", "Write function max_consecutive_ones(nums) finding max consecutive 1s in binary array."),
    ("e_arr_15", "easy", "array", "Write function find_disappeared(nums) finding all numbers missing from 1..n in array."),
    ("e_arr_16", "easy", "array", "Write function sorted_squares(nums) returning squares of sorted array in sorted order."),
    ("e_arr_17", "easy", "array", "Write function valid_mountain_array(arr) checking if array is valid mountain."),
    ("e_arr_18", "easy", "array", "Write function replace_elements(arr) replacing each element with greatest element to its right."),
    ("e_arr_19", "easy", "array", "Write function sort_array_by_parity(arr) moving all even numbers before odd numbers."),
    ("e_arr_20", "easy", "array", "Write function height_checker(heights) counting students not standing in correct positions."),
    ("e_arr_21", "easy", "array", "Write function third_max(nums) returning third maximum distinct element, or max if fewer than 3."),
    ("e_arr_22", "easy", "array", "Write function sum_range(nums, left, right) with prefix sums for range sum queries."),
    ("e_arr_23", "easy", "array", "Write function pivot_index(nums) finding index where left sum equals right sum."),
    ("e_arr_24", "easy", "array", "Write function dominant_index(nums) returning index of largest element at least twice all others."),
    ("e_arr_25", "easy", "array", "Write function diagonal_sum(mat) returning sum of primary and secondary diagonal."),
    ("m_arr_01", "medium", "array", "Write function product_except_self(nums) returning product of all elements except self, O(n)."),
    ("m_arr_02", "medium", "array", "Write function rotate(nums, k) rotating array right by k steps in-place O(1) space."),
    ("m_arr_03", "medium", "array", "Write function search(nums, target) searching in rotated sorted array O(log n)."),
    ("m_arr_04", "medium", "array", "Write function search_range(nums, target) returning first and last position O(log n)."),
    ("m_arr_05", "medium", "array", "Write function three_sum(nums) finding all triplets summing to zero without duplicates."),
    ("m_arr_06", "medium", "array", "Write function next_permutation(nums) finding next lexicographically greater permutation in-place."),
    ("m_arr_07", "medium", "array", "Write function sort_colors(nums) sorting 0,1,2 array in-place Dutch flag algorithm."),
    ("m_arr_08", "medium", "array", "Write function spiral_order(matrix) returning matrix elements in spiral order."),
    ("m_arr_09", "medium", "array", "Write function set_zeroes(matrix) setting row/column to zero if element is zero, O(1) space."),
    ("m_arr_10", "medium", "array", "Write function subarray_sum(nums, k) counting subarrays summing to k using prefix sum."),
    ("m_arr_11", "medium", "array", "Write function top_k_frequent(nums, k) returning k most frequent elements."),
    ("m_arr_12", "medium", "array", "Write function find_peak_element(nums) finding peak element in O(log n)."),
    ("m_arr_13", "medium", "array", "Write function find_min_rotated(nums) finding minimum in rotated sorted array."),
    ("m_arr_14", "medium", "array", "Write function four_sum(nums, target) finding all quadruplets summing to target."),
    ("m_arr_15", "medium", "array", "Write function jump_game(nums) checking if can reach last index."),
    ("m_arr_16", "medium", "array", "Write function merge_intervals(intervals) merging all overlapping intervals."),
    ("m_arr_17", "medium", "array", "Write function insert_interval(intervals, newInterval) inserting and merging interval."),
    ("m_arr_18", "medium", "array", "Write function daily_temperatures(temps) returning days until warmer temperature for each day."),
    ("m_arr_19", "medium", "array", "Write function max_area(height) finding container with most water, two pointer."),
    ("m_arr_20", "medium", "array", "Write function combination_sum(candidates, target) finding all combinations summing to target."),
    ("m_arr_21", "medium", "array", "Write function permutations(nums) returning all possible permutations."),
    ("m_arr_22", "medium", "array", "Write function subsets(nums) returning all possible subsets of array."),
    ("m_arr_23", "medium", "array", "Write function max_sliding_window(nums, k) returning max in each sliding window using deque O(n)."),
    ("m_arr_24", "medium", "array", "Write function min_size_subarray_sum(target, nums) finding minimal length subarray with sum >= target."),
    ("m_arr_25", "medium", "array", "Write function random_pick_weight(w) implementing random index proportional to weight."),
    # === Linked List (30) ===
    ("e_ll_01", "easy", "linked_list", "Write function reverse_list(head) reversing singly linked list iteratively with ListNode class."),
    ("e_ll_02", "easy", "linked_list", "Write function middle_node(head) returning middle node using slow/fast pointers."),
    ("e_ll_03", "easy", "linked_list", "Write function has_cycle(head) detecting cycle using Floyd's tortoise and hare."),
    ("e_ll_04", "easy", "linked_list", "Write function merge_two_lists(l1, l2) merging two sorted linked lists."),
    ("e_ll_05", "easy", "linked_list", "Write function remove_elements(head, val) removing all nodes with given value."),
    ("e_ll_06", "easy", "linked_list", "Write function delete_node(node) deleting node given only access to that node."),
    ("e_ll_07", "easy", "linked_list", "Write function get_decimal_value(head) converting binary linked list to decimal."),
    ("e_ll_08", "easy", "linked_list", "Write function reverse_list_recursive(head) reversing linked list recursively."),
    ("e_ll_09", "easy", "linked_list", "Write function is_palindrome_ll(head) checking if linked list is palindrome."),
    ("e_ll_10", "easy", "linked_list", "Write function remove_duplicates_ll(head) removing duplicates from sorted linked list."),
    ("e_ll_11", "easy", "linked_list", "Write function convert_to_array(head) converting linked list to Python list."),
    ("e_ll_12", "easy", "linked_list", "Write function create_linked_list(values) creating linked list from Python list."),
    ("e_ll_13", "easy", "linked_list", "Write function linked_list_length(head) counting nodes in linked list iteratively."),
    ("e_ll_14", "easy", "linked_list", "Write function search_linked_list(head, target) returning True if target exists in linked list."),
    ("e_ll_15", "easy", "linked_list", "Write function insert_sorted(head, val) inserting value into sorted linked list."),
    ("m_ll_01", "medium", "linked_list", "Write function remove_nth_from_end(head, n) removing n-th node from end in one pass."),
    ("m_ll_02", "medium", "linked_list", "Write function add_two_numbers(l1, l2) adding numbers represented by reversed linked lists."),
    ("m_ll_03", "medium", "linked_list", "Write function get_intersection_node(headA, headB) finding intersection point of two lists."),
    ("m_ll_04", "medium", "linked_list", "Write function odd_even_list(head) grouping odd-indexed before even-indexed nodes."),
    ("m_ll_05", "medium", "linked_list", "Write function swap_pairs(head) swapping every two adjacent nodes."),
    ("m_ll_06", "medium", "linked_list", "Write function rotate_list(head, k) rotating linked list to the right by k places."),
    ("m_ll_07", "medium", "linked_list", "Write function partition_list(head, x) partitioning list so nodes < x come before nodes >= x."),
    ("m_ll_08", "medium", "linked_list", "Write function reverse_between(head, left, right) reversing sublist from position left to right."),
    ("m_ll_09", "medium", "linked_list", "Write function reorder_list(head) reordering L0->Ln->L1->Ln-1... in-place."),
    ("m_ll_10", "medium", "linked_list", "Write function copy_random_list(head) deep copying linked list with random pointers."),
    ("m_ll_11", "medium", "linked_list", "Write function flatten_multilevel(head) flattening multilevel doubly linked list."),
    ("m_ll_12", "medium", "linked_list", "Write function merge_k_lists(lists) merging k sorted linked lists using divide and conquer."),
    ("m_ll_13", "medium", "linked_list", "Write function lru_cache(capacity) implementing LRU cache with doubly linked list + hash map."),
    ("m_ll_14", "medium", "linked_list", "Write function sort_list(head) sorting linked list in O(n log n) using merge sort."),
    ("m_ll_15", "medium", "linked_list", "Write function detect_cycle_start(head) finding where cycle begins in linked list."),
    # === Tree (40) ===
    ("e_tree_01", "easy", "tree", "Write function max_depth(root) returning maximum depth of binary tree, recursive and iterative."),
    ("e_tree_02", "easy", "tree", "Write function invert_tree(root) inverting binary tree (mirror) with TreeNode class."),
    ("e_tree_03", "easy", "tree", "Write function is_same_tree(p, q) checking if two binary trees are identical."),
    ("e_tree_04", "easy", "tree", "Write function is_symmetric(root) checking if binary tree is mirror of itself."),
    ("e_tree_05", "easy", "tree", "Write function has_path_sum(root, targetSum) checking if root-to-leaf path sums to target."),
    ("e_tree_06", "easy", "tree", "Write function preorder_traversal(root) returning preorder traversal iteratively using stack."),
    ("e_tree_07", "easy", "tree", "Write function inorder_traversal(root) returning inorder traversal iteratively."),
    ("e_tree_08", "easy", "tree", "Write function postorder_traversal(root) returning postorder traversal iteratively."),
    ("e_tree_09", "easy", "tree", "Write function min_depth(root) returning minimum depth of binary tree."),
    ("e_tree_10", "easy", "tree", "Write function count_nodes(root) counting total nodes in complete binary tree in O(log^2 n)."),
    ("e_tree_11", "easy", "tree", "Write function is_balanced(root) checking if binary tree is height-balanced."),
    ("e_tree_12", "easy", "tree", "Write function diameter_of_tree(root) returning diameter (longest path between any two nodes)."),
    ("e_tree_13", "easy", "tree", "Write function path_sum(root, target) counting number of paths that sum to target."),
    ("e_tree_14", "easy", "tree", "Write function merge_trees(t1, t2) merging two binary trees by overlapping."),
    ("e_tree_15", "easy", "tree", "Write function search_bst(root, val) searching value in binary search tree."),
    ("e_tree_16", "easy", "tree", "Write function insert_into_bst(root, val) inserting value into BST."),
    ("e_tree_17", "easy", "tree", "Write function is_subtree(s, t) checking if t is subtree of s."),
    ("e_tree_18", "easy", "tree", "Write function leaf_similar(root1, root2) checking if two trees have same leaf value sequence."),
    ("e_tree_19", "easy", "tree", "Write function range_sum_bst(root, low, high) summing values in BST within range."),
    ("e_tree_20", "easy", "tree", "Write function find_mode(root) finding mode(s) in BST."),
    ("m_tree_01", "medium", "tree", "Write function level_order(root) returning binary tree level-order traversal."),
    ("m_tree_02", "medium", "tree", "Write function is_valid_bst(root) validating binary search tree with min/max bounds."),
    ("m_tree_03", "medium", "tree", "Write function lowest_common_ancestor(root, p, q) finding LCA in binary tree."),
    ("m_tree_04", "medium", "tree", "Write function right_side_view(root) returning values visible from right side."),
    ("m_tree_05", "medium", "tree", "Write function build_tree(preorder, inorder) constructing tree from traversal arrays."),
    ("m_tree_06", "medium", "tree", "Write function kth_smallest(root, k) finding k-th smallest in BST."),
    ("m_tree_07", "medium", "tree", "Write function zigzag_level_order(root) returning zigzag level order traversal."),
    ("m_tree_08", "medium", "tree", "Write function flatten_tree(root) flattening binary tree to linked list in-place."),
    ("m_tree_09", "medium", "tree", "Write function populating_next_right(root) connecting each node to its right neighbor."),
    ("m_tree_10", "medium", "tree", "Write function path_sum_iii(root, targetSum) counting paths summing to target (any start/end)."),
    ("m_tree_11", "medium", "tree", "Write function sum_numbers(root) summing all root-to-leaf numbers."),
    ("m_tree_12", "medium", "tree", "Write function binary_tree_paths(root) returning all root-to-leaf paths as strings."),
    ("m_tree_13", "medium", "tree", "Write function sorted_array_to_bst(nums) converting sorted array to balanced BST."),
    ("m_tree_14", "medium", "tree", "Write function preorder_to_bst(preorder) constructing BST from preorder traversal."),
    ("m_tree_15", "medium", "tree", "Write function serialize(root) and deserialize(data) for binary tree with null markers."),
    ("m_tree_16", "medium", "tree", "Write function vertical_order(root) returning vertical order traversal of binary tree."),
    ("m_tree_17", "medium", "tree", "Write function left_side_view(root) returning values visible from left side of binary tree."),
    ("m_tree_18", "medium", "tree", "Write function cousin_nodes(root, x, y) checking if two nodes are cousins (same depth, different parent)."),
    ("m_tree_19", "medium", "tree", "Write function bst_to_greater_tree(root) converting BST so each key is original plus sum of greater keys."),
    ("m_tree_20", "medium", "tree", "Write function max_width_binary_tree(root) returning maximum width of binary tree."),
    # === DP (35) ===
    ("m_dp_01", "medium", "dp", "Write function coin_change(coins, amount) returning fewest coins to make amount."),
    ("m_dp_02", "medium", "dp", "Write function rob(nums) returning max money from non-adjacent houses."),
    ("m_dp_03", "medium", "dp", "Write function length_of_lis(nums) returning length of longest increasing subsequence."),
    ("m_dp_04", "medium", "dp", "Write function unique_paths(m, n) counting paths from top-left to bottom-right."),
    ("m_dp_05", "medium", "dp", "Write function min_distance(word1, word2) computing minimum edit distance."),
    ("m_dp_06", "medium", "dp", "Write function longest_common_subsequence(text1, text2) returning LCS length."),
    ("m_dp_07", "medium", "dp", "Write function max_product(nums) returning max product subarray handling negatives."),
    ("m_dp_08", "medium", "dp", "Write function num_decodings(s) counting ways to decode digit string to letters."),
    ("m_dp_09", "medium", "dp", "Write function can_partition(nums) checking if array can be partitioned into two equal-sum subsets."),
    ("m_dp_10", "medium", "dp", "Write function target_sum(nums, target) counting ways to assign +/- to reach target."),
    ("m_dp_11", "medium", "dp", "Write function find_max_form(strs, m, n) finding max strings with at most m zeros and n ones."),
    ("m_dp_12", "medium", "dp", "Write function knapsack_01(weights, values, capacity) solving 0/1 knapsack problem."),
    ("m_dp_13", "medium", "dp", "Write function longest_palindrome_subseq(s) finding longest palindromic subsequence length."),
    ("m_dp_14", "medium", "dp", "Write function min_cost_climbing_stairs(cost) finding min cost to reach top of stairs."),
    ("m_dp_15", "medium", "dp", "Write function partition_equal_subset(nums) checking if can partition into two equal subsets."),
    ("m_dp_16", "medium", "dp", "Write function change(amount, coins) counting number of ways to make change."),
    ("m_dp_17", "medium", "dp", "Write function max_length_concat(s) finding max length of concatenated string with unique chars."),
    ("m_dp_18", "medium", "dp", "Write function stone_game(piles) determining if first player can win with optimal play."),
    ("m_dp_19", "medium", "dp", "Write function min_falling_path(matrix) finding minimum sum falling path through matrix."),
    ("m_dp_20", "medium", "dp", "Write function delete_and_earn(nums) maximizing points by deleting elements."),
    ("h_dp_01", "hard", "dp", "Write function matrix_chain_order(p) returning minimum multiplications and optimal parenthesization."),
    ("h_dp_02", "hard", "dp", "Write function max_coins(nums) returning max coins from bursting balloons O(n^3)."),
    ("h_dp_03", "hard", "dp", "Write function regular_expression_match(s, p) implementing regex with * and ? using DP."),
    ("h_dp_04", "hard", "dp", "Write function longest_valid_parentheses_dp(s) finding longest valid parentheses substring using DP."),
    ("h_dp_05", "hard", "dp", "Write function maximal_rectangle(matrix) finding maximal rectangle of 1s in binary matrix."),
    ("h_dp_06", "hard", "dp", "Write function palindrome_partitioning_min(s) finding min cuts for palindrome partitioning."),
    ("h_dp_07", "hard", "dp", "Write function distinct_subsequences(s, t) counting distinct subsequences of s equal to t."),
    ("h_dp_08", "hard", "dp", "Write function interleaving_string(s1, s2, s3) checking if s3 is interleaving of s1 and s2."),
    ("h_dp_09", "hard", "dp", "Write function russian_doll_envelopes(envelopes) finding max envelopes that fit inside each other."),
    ("h_dp_10", "hard", "dp", "Write function super_egg_drop(k, n) finding min drops to find critical floor with k eggs."),
    ("h_dp_11", "hard", "dp", "Write function number_of_ways(n, x) counting ways to express n as sum of x-th powers."),
    ("h_dp_12", "hard", "dp", "Write function max_profit_k_transactions(prices, k) finding max profit with at most k transactions."),
    ("h_dp_13", "hard", "dp", "Write function minimum_refuel_stops(target, startFuel, stations) finding min refueling stops."),
    ("h_dp_14", "hard", "dp", "Write function strange_printer(s) finding min turns to print string with strange printer."),
    ("h_dp_15", "hard", "dp", "Write function scramble_string(s1, s2) checking if s2 is scrambled version of s1, DP."),
    # === Graph (25) ===
    ("m_graph_01", "medium", "graph", "Write function num_islands(grid) counting islands in 2D grid using DFS."),
    ("m_graph_02", "medium", "graph", "Write function can_finish(numCourses, prerequisites) detecting cycle via topological sort."),
    ("m_graph_03", "medium", "graph", "Write function clone_graph(node) deep copying graph using BFS with hash map."),
    ("m_graph_04", "medium", "graph", "Write function exist(board, word) searching word in 2D board using DFS backtracking."),
    ("m_graph_05", "medium", "graph", "Write function pacific_atlantic(heights) finding cells that can flow to both oceans."),
    ("m_graph_06", "medium", "graph", "Write function rotting_oranges(grid) returning minutes until all oranges rot, BFS."),
    ("m_graph_07", "medium", "graph", "Write function walls_and_gates(rooms) filling gates with distance to nearest gate, BFS."),
    ("m_graph_08", "medium", "graph", "Write function graph_valid_tree(n, edges) checking if edges form a valid tree."),
    ("m_graph_09", "medium", "graph", "Write function find_redundant_directed(edges) finding redundant directed edge causing cycle."),
    ("m_graph_10", "medium", "graph", "Write function shortest_path_binary_matrix(grid) finding shortest clear path in binary matrix."),
    ("h_graph_01", "hard", "graph", "Write function shortest_path(graph, start, end) using Dijkstra with heap priority queue."),
    ("h_graph_02", "hard", "graph", "Write function top_sort(num_nodes, edges) returning topological order using Kahn's BFS."),
    ("h_graph_03", "hard", "graph", "Write function network_delay_time(times, n, k) finding time for all nodes to receive signal."),
    ("h_graph_04", "hard", "graph", "Write function min_cost_connect_points(points) finding min cost to connect all points (Prim's)."),
    ("h_graph_05", "hard", "graph", "Write function critical_connections(n, connections) finding all bridges in graph."),
    ("h_graph_06", "hard", "graph", "Write function accounts_merge(accounts) merging accounts with common emails using Union-Find."),
    ("h_graph_07", "hard", "graph", "Write function bellman_ford(edges, n, src) finding shortest paths with negative edges."),
    ("h_graph_08", "hard", "graph", "Write function strongly_connected(n, edges) finding SCCs using Kosaraju's algorithm."),
    ("h_graph_09", "hard", "graph", "Write function max_flow(graph, source, sink) computing max flow using Ford-Fulkerson."),
    ("h_graph_10", "hard", "graph", "Write function eulerian_path(n, edges) finding Eulerian path in directed graph."),
    ("h_graph_11", "hard", "graph", "Write function bipartite_check(graph) checking if graph is bipartite using BFS coloring."),
    ("h_graph_12", "hard", "graph", "Write function articulation_points(n, edges) finding all articulation points in graph."),
    ("h_graph_13", "hard", "graph", "Write function shortest_path_all_keys(grid) finding shortest path collecting all keys, BFS."),
    ("h_graph_14", "hard", "graph", "Write function cheapest_flights_k_stops(n, flights, src, dst, k) finding cheapest price with k stops."),
    ("h_graph_15", "hard", "graph", "Write function ladder_length(beginWord, endWord, wordList) finding shortest transformation sequence."),
    # === Design (25) ===
    ("m_design_01", "medium", "design", "Write MinStack class with push, pop, top, get_min all O(1) using two stacks."),
    ("m_design_02", "medium", "design", "Write LRUCache class with O(1) get and put using OrderedDict."),
    ("m_design_03", "medium", "design", "Write @timer decorator measuring and printing execution time."),
    ("m_design_04", "medium", "design", "Write @retry(max_attempts=3, delay=1) decorator with exponential backoff."),
    ("m_design_05", "medium", "design", "Write @singleton decorator for thread-safe singleton pattern."),
    ("m_design_06", "medium", "design", "Write Trie class with insert, search, startsWith methods."),
    ("m_design_07", "medium", "design", "Write BSTIterator class that iterates BST in-order with O(h) memory."),
    ("m_design_08", "medium", "design", "Write SnapshotArray class supporting set, snap, and get operations."),
    ("m_design_09", "medium", "design", "Write @memoize decorator caching function results based on arguments."),
    ("m_design_10", "medium", "design", "Write LinkedListDeque class with add_front, add_rear, remove_front, remove_rear O(1)."),
    ("h_design_01", "hard", "design", "Write ConnectionPool class with acquire(timeout) and release, thread-safe, max connections."),
    ("h_design_02", "hard", "design", "Write RateLimiter class with token bucket algorithm, thread-safe."),
    ("h_design_03", "hard", "design", "Write EventEmitter class with on(event, handler), emit(event, *args), off(event, handler)."),
    ("h_design_04", "hard", "design", "Write PubSub class with subscribe, publish, unsubscribe methods."),
    ("h_design_05", "hard", "design", "Write generic ObjectPool class with borrow() and return(obj), lazy init, max size."),
    ("h_design_06", "hard", "design", "Implement Observer pattern: Subject with attach, detach, notify observers."),
    ("h_design_07", "hard", "design", "Write Scheduler class with schedule(task, delay), repeating(task, interval), cancel(task_id)."),
    ("h_design_08", "hard", "design", "Write generic Graph class with add_edge, bfs, dfs, shortest_path, has_cycle methods."),
    ("h_design_09", "hard", "design", "Write TransactionManager class with begin, commit, rollback supporting nested transactions."),
    ("h_design_10", "hard", "design", "Write CacheAside class with get(key), put(key, value, ttl), eviction policy (LRU/LFU)."),
    ("h_design_11", "hard", "design", "Write MessageQueue class with enqueue, dequeue, dead_letter_queue, retry logic."),
    ("h_design_12", "hard", "design", "Write TaskScheduler with dependencies: schedule(task, deps), execute_all(), topological order."),
    ("h_design_13", "hard", "design", "Write generic SortedContainer class with add, remove, floor, ceiling, rank operations."),
    ("h_design_14", "hard", "design", "Write UndoRedoManager class supporting execute, undo, redo, clear operations."),
    ("h_design_15", "hard", "design", "Write DiningPhilosophers class solving dining philosophers problem with deadlock prevention."),
    # === Algorithm (20) ===
    ("h_algo_01", "hard", "algorithm", "Write function find_median_sorted_arrays(nums1, nums2) finding median in O(log(min(m,n)))."),
    ("h_algo_02", "hard", "algorithm", "Write function merge_k_lists(lists) merging k sorted linked lists using min-heap O(n log k)."),
    ("h_algo_03", "hard", "algorithm", "Write function trap(height) computing trapped rainwater using two pointers O(n)."),
    ("h_algo_04", "hard", "algorithm", "Write function largest_rectangle_area(heights) computing largest rectangle in histogram using stack."),
    ("h_algo_05", "hard", "algorithm", "Write function max_sliding_window(nums, k) returning max in each window using deque O(n)."),
    ("h_algo_06", "hard", "algorithm", "Write function alien_order(words) finding alien language letter order via topological sort."),
    ("h_algo_07", "hard", "algorithm", "Write function ladder_length(beginWord, endWord, wordList) finding shortest transformation sequence."),
    ("h_algo_08", "hard", "algorithm", "Write function solve_n_queens(n) returning all N-Queens solutions using backtracking."),
    ("h_algo_09", "hard", "algorithm", "Write function is_match(s, p) implementing regex matching with . and * using DP."),
    ("h_algo_10", "hard", "algorithm", "Write function shortest_palindrome(s) finding shortest palindrome by adding chars in front."),
    ("h_algo_11", "hard", "algorithm", "Write function min_window_subsequence(s1, s2) finding minimum window in s1 containing s2 as subsequence."),
    ("h_algo_12", "hard", "algorithm", "Write function count_smaller(nums) counting smaller elements to the right using merge sort."),
    ("h_algo_13", "hard", "algorithm", "Write function expression_add_operators(num, target) adding +,-,* to reach target value."),
    ("h_algo_14", "hard", "algorithm", "Write function is_number(s) validating if string is valid number (scientific notation)."),
    ("h_algo_15", "hard", "algorithm", "Write function max_points_on_line(points) finding max points on same straight line."),
    ("h_algo_16", "hard", "algorithm", "Write function skyline(buildings) computing skyline silhouette from building list."),
    ("h_algo_17", "hard", "algorithm", "Write function min_insertions_palindrome(s) finding min insertions to make string palindrome."),
    ("h_algo_18", "hard", "algorithm", "Write function max_sum_rectangle(matrix, k) finding max sum rectangle with sum <= k."),
    ("h_algo_19", "hard", "algorithm", "Write function strange_printer(s) finding min turns to print string with strange printer."),
    ("h_algo_20", "hard", "algorithm", "Write function scramble_string(s1, s2) checking if s2 is scrambled version of s1."),
    # === Math (20) ===
    ("e_math_01", "easy", "math", "Write function is_palindrome(x) checking if integer is palindrome without converting to string."),
    ("e_math_02", "easy", "math", "Write function fizzbuzz(n) returning list of strings for 1..n with FizzBuzz rules."),
    ("e_math_03", "easy", "math", "Write function is_prime(n) returning True if n is prime in O(sqrt(n))."),
    ("e_math_04", "easy", "math", "Write function gcd(a, b) using Euclidean algorithm for greatest common divisor."),
    ("e_math_05", "easy", "math", "Write function is_power_of_two(n) returning True using bit manipulation."),
    ("e_math_06", "easy", "math", "Write function factorial(n) computing n! both recursively and iteratively."),
    ("e_math_07", "easy", "math", "Write function count_digits(n) counting digits in integer without converting to string."),
    ("e_math_08", "easy", "math", "Write function sum_of_digits(n) returning sum of all digits in integer."),
    ("e_math_09", "easy", "math", "Write function is_perfect_square(n) checking if n is perfect square without sqrt."),
    ("e_math_10", "easy", "math", "Write function climb_stairs(n) returning number of ways to climb n stairs taking 1 or 2 steps."),
    ("e_math_11", "easy", "math", "Write function fib(n) returning n-th Fibonacci number with O(n) time O(1) space."),
    ("e_math_12", "easy", "math", "Write function roman_to_int(s) converting Roman numeral string to integer."),
    ("e_math_13", "easy", "math", "Write function add_binary(a, b) adding two binary strings and returning binary string."),
    ("e_math_14", "easy", "math", "Write function my_sqrt(x) computing integer square root without built-in sqrt."),
    ("e_math_15", "easy", "math", "Write function trailing_zeroes(n) counting trailing zeros in n!."),
    ("e_math_16", "easy", "math", "Write function excel_column(n) converting integer to Excel column title (1->A, 27->AA)."),
    ("e_math_17", "easy", "math", "Write function happy_number(n) checking if number eventually reaches 1 by summing squares of digits."),
    ("e_math_18", "easy", "math", "Write function count_primes(n) counting primes less than n using Sieve of Eratosthenes."),
    ("e_math_19", "easy", "math", "Write function power(x, n) computing x^n efficiently using fast exponentiation."),
    ("e_math_20", "easy", "math", "Write function max_69_number(num) returning max number by changing at most one digit 6 to 9."),
    # === Concurrency (15) ===
    ("h_conc_01", "hard", "concurrency", "Write thread-safe Counter with increment() and get() using Lock. Test with 10 threads."),
    ("h_conc_02", "hard", "concurrency", "Implement ReadWriteLock allowing concurrent reads but exclusive writes using threading."),
    ("h_conc_03", "hard", "concurrency", "Write ThreadPool class with submit(task) and map(tasks), fixed workers, queue-based."),
    ("h_conc_04", "hard", "concurrency", "Write async fetch_all(urls) fetching multiple URLs concurrently using asyncio."),
    ("h_conc_05", "hard", "concurrency", "Write thread-safe producer-consumer using threading + Queue with multiple producers/consumers."),
    ("h_conc_06", "hard", "concurrency", "Implement Barrier class with wait(timeout) using threading.Condition."),
    ("h_conc_07", "hard", "concurrency", "Write async rate_limiter(n, period) limiting concurrent requests using asyncio.Semaphore."),
    ("h_conc_08", "hard", "concurrency", "Write DiningPhilosophers class solving dining philosophers problem with deadlock prevention."),
    ("h_conc_09", "hard", "concurrency", "Implement CountDownLatch with count_down() and await() using threading primitives."),
    ("h_conc_10", "hard", "concurrency", "Write async pipeline(stages) chaining async processing stages with bounded queues."),
    ("h_conc_11", "hard", "concurrency", "Write thread-safe CircularBuffer class with put, get, and capacity limit."),
    ("h_conc_12", "hard", "concurrency", "Implement Cancellation token system: CancellationToken with cancel/is_cancelled/callbacks."),
    ("h_conc_13", "hard", "concurrency", "Write async bounded executor(max_workers) limiting concurrent coroutine execution."),
    ("h_conc_14", "hard", "concurrency", "Write thread-safe PriorityBlockingQueue with put, get (blocks when empty), peek."),
    ("h_conc_15", "hard", "concurrency", "Implement ReadWriteLock with writer preference using threading.Condition."),
]


# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = """You are an expert Python programmer. Solve problems in 4 steps:
1. UNDERSTAND - brief analysis of inputs, outputs, constraints
2. DESIGN - brief plan: data structures, algorithm, complexity
3. IMPLEMENT - complete Python code in a code block
4. VERIFY - test with 1-2 examples

Rules:
- Python only. Standard library only (collections, heapq, bisect, itertools, functools, typing, math).
- Code must be complete and runnable. No TODOs, no placeholders.
- Be concise. Focus on correctness."""


# ============================================================
# 质量检测器
# ============================================================
def detect_banned_imports(code: str) -> list:
    """检测禁止调用的库."""
    found = []
    for ban in BANNED_IMPORTS:
        if re.search(rf'\bimport\s+{re.escape(ban)}\b', code, re.IGNORECASE):
            found.append(ban)
        if re.search(rf'\bfrom\s+{re.escape(ban)}\b', code, re.IGNORECASE):
            found.append(ban)
    return found


def detect_language_error(text: str) -> bool:
    """检测非 Python 语言."""
    java_patterns = [r'\bpublic\s+class\b', r'\bSystem\.out\b', r'\bvoid\s+\w+\(']
    cpp_patterns = [r'\b#include\b', r'\bstd::', r'\bcout\b']
    js_patterns = [r'\bconsole\.log\b', r'\blet\s+\w+\s*=', r'\bconst\s+\w+\s*=\s*\{']
    for p in java_patterns + cpp_patterns + js_patterns:
        if re.search(p, text):
            return True
    return False


def detect_missing_code(text: str) -> bool:
    """检测是否缺少代码."""
    return 'def ' not in text and 'class ' not in text


def detect_truncation(text: str) -> bool:
    """检测输出截断."""
    truncation_signals = [
        text.endswith('...'),
        text.endswith('..'),
        len(text) < 50,
        # 话说到一半
        re.search(r'(however|but|so|therefore|thus)\s*$', text, re.IGNORECASE),
    ]
    return any(truncation_signals)


def detect_fake_reasoning(text: str) -> bool:
    """检测假推理 (只有结论没有过程)."""
    has_step = bool(re.search(r'(Step\s*\d|UNDERSTAND|DESIGN|IMPLEMENT|VERIFY)', text, re.IGNORECASE))
    has_analysis = bool(re.search(r'(because|since|therefore|complexity|edge case|constraint|O\(|time|space|algorithm)', text, re.IGNORECASE))
    has_explanation = bool(re.search(r'(We |This |The |I |First|Then|Next|Finally|Approach|Strategy|Plan)', text))
    return not (has_step or has_analysis or has_explanation)


def detect_template_response(text: str) -> bool:
    """检测模板化回答."""
    templates = [
        r'^(Sure|Certainly|Of course|Here is|Below is)',
        r'I\'d be happy to',
        r'Let me (help|show|explain)',
    ]
    for t in templates:
        if re.search(t, text, re.IGNORECASE):
            return True
    return False


def extract_code(text: str) -> str:
    """提取代码块."""
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def check_syntax(code: str) -> bool:
    """检查 Python 语法."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_completeness(code: str) -> bool:
    """检查代码完整性 (有 def/class, 有 return)."""
    has_def = bool(re.search(r'(def \w+|class \w+)', code))
    has_return = 'return' in code or 'yield' in code
    # 检查是否有 TODO/placeholder
    has_placeholder = bool(re.search(r'(TODO|FIXME|\.\.\.|\.\.\.)', code))
    return has_def and has_return and not has_placeholder


def quality_check(text: str, problem_prompt: str) -> dict:
    """综合质量检测."""
    code = extract_code(text)
    has_code = 'def ' in code or 'class ' in code
    has_code_block = '```' in text

    checks = {
        "has_code": has_code,
        "has_code_block": has_code_block,
        "syntax_ok": check_syntax(code) if has_code else False,
        "completeness_ok": check_completeness(code) if has_code else False,
        "no_banned_imports": len(detect_banned_imports(code)) == 0,
        "language_ok": not detect_language_error(text),
        "no_truncation": not detect_truncation(text),
        "no_template": not detect_template_response(text),
        "banned_found": detect_banned_imports(code),
    }
    # 核心要求: 有代码 + 语法正确 + 完整 + 无禁止库 + 语言正确
    core_ok = checks["has_code"] and checks["syntax_ok"] and checks["completeness_ok"]
    core_ok = core_ok and checks["no_banned_imports"] and checks["language_ok"]
    checks["pass"] = core_ok
    return checks


# ============================================================
# 推理链提取
# ============================================================
def extract_reasoning(text: str) -> str:
    """提取推理部分 (Step 1-2 的内容)."""
    # 找到 Step 1 到 Step 3 之间的内容
    m = re.search(r'(Step\s*1.*?)(?=Step\s*3|```)', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 退而求其次: 取代码块之前的内容
    m = re.search(r'(.*?)(?=```)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_response(text: str) -> str:
    """提取代码实现部分 (Step 3 的代码)."""
    code = extract_code(text)
    # 加上 Step 4 验证部分
    m = re.search(r'(Step\s*4.*?)(?=$)', text, re.DOTALL | re.IGNORECASE)
    verify = m.group(1).strip() if m else ""
    if verify:
        return f"```python\n{code}\n```\n\n{verify}"
    return f"```python\n{code}\n```"


# ============================================================
# 主流程
# ============================================================
def load_model(path):
    print(f"加载模型: {path}")
    t0 = time.time()
    llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=8192, n_threads=8,
                n_batch=512, use_mmap=True, verbose=False, chat_format="chatml")
    print(f"  耗时: {time.time()-t0:.1f}s")
    return llm


def generate(llm, prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    output = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
    text = output["choices"][0]["message"]["content"]

    # 如果没有代码, 用更简单的 prompt 重试一次
    if 'def ' not in text and 'class ' not in text:
        simple_messages = [
            {"role": "system", "content": "Write Python code only. No explanation."},
            {"role": "user", "content": prompt},
        ]
        output = llm.create_chat_completion(messages=simple_messages, max_tokens=512, temperature=0.1)
        text = output["choices"][0]["message"]["content"]

    return text


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "stats": {"total": 0, "passed": 0, "failed": 0, "reasons": {}}}


def save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-problems", type=int, default=len(PROBLEMS))
    args = parser.parse_args()

    llm = load_model(args.model)
    ckpt = load_checkpoint() if args.resume else {"completed": [], "stats": {"total": 0, "passed": 0, "failed": 0, "reasons": {}}}

    completed_set = set(ckpt["completed"])
    out_f = open(OUTPUT_FILE, "a" if args.resume else "w", encoding="utf-8")

    problems = PROBLEMS[:args.max_problems]
    total = len(problems)
    passed = ckpt["stats"]["passed"]
    failed = ckpt["stats"]["failed"]

    print(f"\n{'='*60}")
    print(f"构建高质量编程推理数据集")
    print(f"模型: {args.model}")
    print(f"问题数: {total} | 已完成: {len(completed_set)} | 通过: {passed} | 失败: {failed}")
    print(f"{'='*60}\n")

    for i, (pid, diff, cat, prompt) in enumerate(problems):
        if pid in completed_set:
            continue

        print(f"[{i+1}/{total}] [{diff}] [{cat}] {pid}")

        # 生成
        t0 = time.time()
        try:
            response = generate(llm, prompt)
        except Exception as e:
            print(f"  ERROR: {e}")
            ckpt["completed"].append(pid)
            ckpt["stats"]["failed"] += 1
            ckpt["stats"]["reasons"][pid] = f"generate_error: {str(e)[:50]}"
            save_checkpoint(ckpt)
            continue
        elapsed = time.time() - t0

        # 质量检测
        checks = quality_check(response, prompt)

        if checks["pass"]:
            # 通过: 提取推理和代码, 保存
            reasoning = extract_reasoning(response)
            code_response = extract_response(response)

            entry = {
                "instruction": prompt,
                "reasoning": reasoning,
                "response": code_response,
                "metadata": {
                    "problem_id": pid,
                    "difficulty": diff,
                    "category": cat,
                    "gen_time": round(elapsed, 1),
                }
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            passed += 1
            print(f"  PASS ({elapsed:.1f}s) | reasoning: {len(reasoning)} chars | code: {len(code_response)} chars")
        else:
            # 失败: 记录原因
            failed += 1
            reasons = [k for k, v in checks.items() if k not in ("pass", "banned_found") and not v]
            if checks.get("banned_found"):
                reasons.append(f"banned={checks['banned_found']}")
            ckpt["stats"]["reasons"][pid] = "; ".join(reasons)
            print(f"  FAIL ({elapsed:.1f}s) | {', '.join(reasons)}")

        ckpt["completed"].append(pid)
        ckpt["stats"]["total"] = i + 1
        ckpt["stats"]["passed"] = passed
        ckpt["stats"]["failed"] = failed

        # 每 10 题保存 checkpoint
        if (i + 1) % 10 == 0:
            save_checkpoint(ckpt)

    save_checkpoint(ckpt)
    out_f.close()

    print(f"\n{'='*60}")
    print(f"数据集构建完成!")
    print(f"通过: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"失败: {failed}/{total}")
    print(f"输出: {OUTPUT_FILE}")
    print(f"{'='*60}")

    # 打印失败原因统计
    if ckpt["stats"]["reasons"]:
        print(f"\n--- 失败原因统计 ---")
        reason_counts = {}
        for pid, reason in ckpt["stats"]["reasons"].items():
            for r in reason.split("; "):
                reason_counts[r] = reason_counts.get(r, 0) + 1
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {r}: {c}")


if __name__ == "__main__":
    main()
