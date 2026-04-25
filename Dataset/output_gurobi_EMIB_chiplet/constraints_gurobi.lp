\ Model ChipletPlacementGrid
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xbe9ceb03ed2fa529
Minimize
  0 r_7 + 0 r_8 + 0 r_15 + 0.00434782608695652 aspect_ratio_penalty
   + 4.41386e-06 wirelength + 0.1101698475762103 bbox_area_proxy_t
Subject To
 width_rotation_0: - r_0 + w_var_0 = 5
 height_rotation_0: r_0 + h_var_0 = 6
 x_grid_var_ub_0: w_var_0 + x_grid_var_0 <= 387
 y_grid_var_ub_0: h_var_0 + y_grid_var_0 <= 387
 width_rotation_1: - 3 r_1 + w_var_1 = 12
 height_rotation_1: 3 r_1 + h_var_1 = 15
 x_grid_var_ub_1: w_var_1 + x_grid_var_1 <= 387
 y_grid_var_ub_1: h_var_1 + y_grid_var_1 <= 387
 width_rotation_2: r_2 + w_var_2 = 26
 height_rotation_2: - r_2 + h_var_2 = 25
 x_grid_var_ub_2: w_var_2 + x_grid_var_2 <= 387
 y_grid_var_ub_2: h_var_2 + y_grid_var_2 <= 387
 width_rotation_3: r_3 + w_var_3 = 29
 height_rotation_3: - r_3 + h_var_3 = 28
 x_grid_var_ub_3: w_var_3 + x_grid_var_3 <= 387
 y_grid_var_ub_3: h_var_3 + y_grid_var_3 <= 387
 width_rotation_4: r_4 + w_var_4 = 29
 height_rotation_4: - r_4 + h_var_4 = 28
 x_grid_var_ub_4: w_var_4 + x_grid_var_4 <= 387
 y_grid_var_ub_4: h_var_4 + y_grid_var_4 <= 387
 width_rotation_5: - r_5 + w_var_5 = 22
 height_rotation_5: r_5 + h_var_5 = 23
 x_grid_var_ub_5: w_var_5 + x_grid_var_5 <= 387
 y_grid_var_ub_5: h_var_5 + y_grid_var_5 <= 387
 width_rotation_6: r_6 + w_var_6 = 29
 height_rotation_6: - r_6 + h_var_6 = 28
 x_grid_var_ub_6: w_var_6 + x_grid_var_6 <= 387
 y_grid_var_ub_6: h_var_6 + y_grid_var_6 <= 387
 width_rotation_7: w_var_7 = 16
 height_rotation_7: h_var_7 = 16
 x_grid_var_ub_7: w_var_7 + x_grid_var_7 <= 387
 y_grid_var_ub_7: h_var_7 + y_grid_var_7 <= 387
 width_rotation_8: w_var_8 = 20
 height_rotation_8: h_var_8 = 20
 x_grid_var_ub_8: w_var_8 + x_grid_var_8 <= 387
 y_grid_var_ub_8: h_var_8 + y_grid_var_8 <= 387
 width_rotation_9: - 2 r_9 + w_var_9 = 25
 height_rotation_9: 2 r_9 + h_var_9 = 27
 x_grid_var_ub_9: w_var_9 + x_grid_var_9 <= 387
 y_grid_var_ub_9: h_var_9 + y_grid_var_9 <= 387
 width_rotation_10: - r_10 + w_var_10 = 19
 height_rotation_10: r_10 + h_var_10 = 20
 x_grid_var_ub_10: w_var_10 + x_grid_var_10 <= 387
 y_grid_var_ub_10: h_var_10 + y_grid_var_10 <= 387
 width_rotation_11: - 3 r_11 + w_var_11 = 25
 height_rotation_11: 3 r_11 + h_var_11 = 28
 x_grid_var_ub_11: w_var_11 + x_grid_var_11 <= 387
 y_grid_var_ub_11: h_var_11 + y_grid_var_11 <= 387
 width_rotation_12: 2 r_12 + w_var_12 = 25
 height_rotation_12: - 2 r_12 + h_var_12 = 23
 x_grid_var_ub_12: w_var_12 + x_grid_var_12 <= 387
 y_grid_var_ub_12: h_var_12 + y_grid_var_12 <= 387
 width_rotation_13: - 3 r_13 + w_var_13 = 19
 height_rotation_13: 3 r_13 + h_var_13 = 22
 x_grid_var_ub_13: w_var_13 + x_grid_var_13 <= 387
 y_grid_var_ub_13: h_var_13 + y_grid_var_13 <= 387
 width_rotation_14: 3 r_14 + w_var_14 = 19
 height_rotation_14: - 3 r_14 + h_var_14 = 16
 x_grid_var_ub_14: w_var_14 + x_grid_var_14 <= 387
 y_grid_var_ub_14: h_var_14 + y_grid_var_14 <= 387
 width_rotation_15: w_var_15 = 27
 height_rotation_15: h_var_15 = 27
 x_grid_var_ub_15: w_var_15 + x_grid_var_15 <= 387
 y_grid_var_ub_15: h_var_15 + y_grid_var_15 <= 387
 cx_def_0: - 0.5 w_var_0 - x_grid_var_0 + cx_grid_var_0 = 0
 cy_def_0: - 0.5 h_var_0 - y_grid_var_0 + cy_grid_var_0 = 0
 cx_def_1: - 0.5 w_var_1 - x_grid_var_1 + cx_grid_var_1 = 0
 cy_def_1: - 0.5 h_var_1 - y_grid_var_1 + cy_grid_var_1 = 0
 cx_def_2: - 0.5 w_var_2 - x_grid_var_2 + cx_grid_var_2 = 0
 cy_def_2: - 0.5 h_var_2 - y_grid_var_2 + cy_grid_var_2 = 0
 cx_def_3: - 0.5 w_var_3 - x_grid_var_3 + cx_grid_var_3 = 0
 cy_def_3: - 0.5 h_var_3 - y_grid_var_3 + cy_grid_var_3 = 0
 cx_def_4: - 0.5 w_var_4 - x_grid_var_4 + cx_grid_var_4 = 0
 cy_def_4: - 0.5 h_var_4 - y_grid_var_4 + cy_grid_var_4 = 0
 cx_def_5: - 0.5 w_var_5 - x_grid_var_5 + cx_grid_var_5 = 0
 cy_def_5: - 0.5 h_var_5 - y_grid_var_5 + cy_grid_var_5 = 0
 cx_def_6: - 0.5 w_var_6 - x_grid_var_6 + cx_grid_var_6 = 0
 cy_def_6: - 0.5 h_var_6 - y_grid_var_6 + cy_grid_var_6 = 0
 cx_def_7: - 0.5 w_var_7 - x_grid_var_7 + cx_grid_var_7 = 0
 cy_def_7: - 0.5 h_var_7 - y_grid_var_7 + cy_grid_var_7 = 0
 cx_def_8: - 0.5 w_var_8 - x_grid_var_8 + cx_grid_var_8 = 0
 cy_def_8: - 0.5 h_var_8 - y_grid_var_8 + cy_grid_var_8 = 0
 cx_def_9: - 0.5 w_var_9 - x_grid_var_9 + cx_grid_var_9 = 0
 cy_def_9: - 0.5 h_var_9 - y_grid_var_9 + cy_grid_var_9 = 0
 cx_def_10: - 0.5 w_var_10 - x_grid_var_10 + cx_grid_var_10 = 0
 cy_def_10: - 0.5 h_var_10 - y_grid_var_10 + cy_grid_var_10 = 0
 cx_def_11: - 0.5 w_var_11 - x_grid_var_11 + cx_grid_var_11 = 0
 cy_def_11: - 0.5 h_var_11 - y_grid_var_11 + cy_grid_var_11 = 0
 cx_def_12: - 0.5 w_var_12 - x_grid_var_12 + cx_grid_var_12 = 0
 cy_def_12: - 0.5 h_var_12 - y_grid_var_12 + cy_grid_var_12 = 0
 cx_def_13: - 0.5 w_var_13 - x_grid_var_13 + cx_grid_var_13 = 0
 cy_def_13: - 0.5 h_var_13 - y_grid_var_13 + cy_grid_var_13 = 0
 cx_def_14: - 0.5 w_var_14 - x_grid_var_14 + cx_grid_var_14 = 0
 cy_def_14: - 0.5 h_var_14 - y_grid_var_14 + cy_grid_var_14 = 0
 cx_def_15: - 0.5 w_var_15 - x_grid_var_15 + cx_grid_var_15 = 0
 cy_def_15: - 0.5 h_var_15 - y_grid_var_15 + cy_grid_var_15 = 0
 non_overlap_any_0_1: p_left_0_1 + p_right_0_1 + p_down_0_1 + p_up_0_1
   >= 1
 non_overlap_left_0_1: w_var_0 + x_grid_var_0 - x_grid_var_1
   + 774 p_left_0_1 <= 774
 non_overlap_left_rev_0_1: - w_var_0 - x_grid_var_0 + x_grid_var_1
   - 774 p_left_0_1 <= 0
 non_overlap_right_0_1: w_var_1 - x_grid_var_0 + x_grid_var_1
   + 774 p_right_0_1 <= 774
 non_overlap_right_rev_0_1: - w_var_1 + x_grid_var_0 - x_grid_var_1
   - 774 p_right_0_1 <= 0
 non_overlap_down_0_1: h_var_0 + y_grid_var_0 - y_grid_var_1
   + 774 p_down_0_1 <= 774
 non_overlap_down_rev_0_1: - h_var_0 - y_grid_var_0 + y_grid_var_1
   - 774 p_down_0_1 <= 0
 non_overlap_up_0_1: h_var_1 - y_grid_var_0 + y_grid_var_1 + 774 p_up_0_1
   <= 774
 non_overlap_up_rev_0_1: - h_var_1 + y_grid_var_0 - y_grid_var_1
   - 774 p_up_0_1 <= 0
 non_overlap_any_0_2: p_left_0_2 + p_right_0_2 + p_down_0_2 + p_up_0_2
   >= 1
 non_overlap_left_0_2: w_var_0 + x_grid_var_0 - x_grid_var_2
   + 774 p_left_0_2 <= 774
 non_overlap_left_rev_0_2: - w_var_0 - x_grid_var_0 + x_grid_var_2
   - 774 p_left_0_2 <= 0
 non_overlap_right_0_2: w_var_2 - x_grid_var_0 + x_grid_var_2
   + 774 p_right_0_2 <= 774
 non_overlap_right_rev_0_2: - w_var_2 + x_grid_var_0 - x_grid_var_2
   - 774 p_right_0_2 <= 0
 non_overlap_down_0_2: h_var_0 + y_grid_var_0 - y_grid_var_2
   + 774 p_down_0_2 <= 774
 non_overlap_down_rev_0_2: - h_var_0 - y_grid_var_0 + y_grid_var_2
   - 774 p_down_0_2 <= 0
 non_overlap_up_0_2: h_var_2 - y_grid_var_0 + y_grid_var_2 + 774 p_up_0_2
   <= 774
 non_overlap_up_rev_0_2: - h_var_2 + y_grid_var_0 - y_grid_var_2
   - 774 p_up_0_2 <= 0
 non_overlap_any_0_3: p_left_0_3 + p_right_0_3 + p_down_0_3 + p_up_0_3
   >= 1
 non_overlap_left_0_3: w_var_0 + x_grid_var_0 - x_grid_var_3
   + 774 p_left_0_3 <= 774
 non_overlap_left_rev_0_3: - w_var_0 - x_grid_var_0 + x_grid_var_3
   - 774 p_left_0_3 <= 0
 non_overlap_right_0_3: w_var_3 - x_grid_var_0 + x_grid_var_3
   + 774 p_right_0_3 <= 774
 non_overlap_right_rev_0_3: - w_var_3 + x_grid_var_0 - x_grid_var_3
   - 774 p_right_0_3 <= 0
 non_overlap_down_0_3: h_var_0 + y_grid_var_0 - y_grid_var_3
   + 774 p_down_0_3 <= 774
 non_overlap_down_rev_0_3: - h_var_0 - y_grid_var_0 + y_grid_var_3
   - 774 p_down_0_3 <= 0
 non_overlap_up_0_3: h_var_3 - y_grid_var_0 + y_grid_var_3 + 774 p_up_0_3
   <= 774
 non_overlap_up_rev_0_3: - h_var_3 + y_grid_var_0 - y_grid_var_3
   - 774 p_up_0_3 <= 0
 non_overlap_any_0_4: p_left_0_4 + p_right_0_4 + p_down_0_4 + p_up_0_4
   >= 1
 non_overlap_left_0_4: w_var_0 + x_grid_var_0 - x_grid_var_4
   + 774 p_left_0_4 <= 774
 non_overlap_left_rev_0_4: - w_var_0 - x_grid_var_0 + x_grid_var_4
   - 774 p_left_0_4 <= 0
 non_overlap_right_0_4: w_var_4 - x_grid_var_0 + x_grid_var_4
   + 774 p_right_0_4 <= 774
 non_overlap_right_rev_0_4: - w_var_4 + x_grid_var_0 - x_grid_var_4
   - 774 p_right_0_4 <= 0
 non_overlap_down_0_4: h_var_0 + y_grid_var_0 - y_grid_var_4
   + 774 p_down_0_4 <= 774
 non_overlap_down_rev_0_4: - h_var_0 - y_grid_var_0 + y_grid_var_4
   - 774 p_down_0_4 <= 0
 non_overlap_up_0_4: h_var_4 - y_grid_var_0 + y_grid_var_4 + 774 p_up_0_4
   <= 774
 non_overlap_up_rev_0_4: - h_var_4 + y_grid_var_0 - y_grid_var_4
   - 774 p_up_0_4 <= 0
 non_overlap_any_0_5: p_left_0_5 + p_right_0_5 + p_down_0_5 + p_up_0_5
   >= 1
 non_overlap_left_0_5: w_var_0 + x_grid_var_0 - x_grid_var_5
   + 774 p_left_0_5 <= 774
 non_overlap_left_rev_0_5: - w_var_0 - x_grid_var_0 + x_grid_var_5
   - 774 p_left_0_5 <= 0
 non_overlap_right_0_5: w_var_5 - x_grid_var_0 + x_grid_var_5
   + 774 p_right_0_5 <= 774
 non_overlap_right_rev_0_5: - w_var_5 + x_grid_var_0 - x_grid_var_5
   - 774 p_right_0_5 <= 0
 non_overlap_down_0_5: h_var_0 + y_grid_var_0 - y_grid_var_5
   + 774 p_down_0_5 <= 774
 non_overlap_down_rev_0_5: - h_var_0 - y_grid_var_0 + y_grid_var_5
   - 774 p_down_0_5 <= 0
 non_overlap_up_0_5: h_var_5 - y_grid_var_0 + y_grid_var_5 + 774 p_up_0_5
   <= 774
 non_overlap_up_rev_0_5: - h_var_5 + y_grid_var_0 - y_grid_var_5
   - 774 p_up_0_5 <= 0
 non_overlap_any_0_6: p_left_0_6 + p_right_0_6 + p_down_0_6 + p_up_0_6
   >= 1
 non_overlap_left_0_6: w_var_0 + x_grid_var_0 - x_grid_var_6
   + 774 p_left_0_6 <= 774
 non_overlap_left_rev_0_6: - w_var_0 - x_grid_var_0 + x_grid_var_6
   - 774 p_left_0_6 <= 0
 non_overlap_right_0_6: w_var_6 - x_grid_var_0 + x_grid_var_6
   + 774 p_right_0_6 <= 774
 non_overlap_right_rev_0_6: - w_var_6 + x_grid_var_0 - x_grid_var_6
   - 774 p_right_0_6 <= 0
 non_overlap_down_0_6: h_var_0 + y_grid_var_0 - y_grid_var_6
   + 774 p_down_0_6 <= 774
 non_overlap_down_rev_0_6: - h_var_0 - y_grid_var_0 + y_grid_var_6
   - 774 p_down_0_6 <= 0
 non_overlap_up_0_6: h_var_6 - y_grid_var_0 + y_grid_var_6 + 774 p_up_0_6
   <= 774
 non_overlap_up_rev_0_6: - h_var_6 + y_grid_var_0 - y_grid_var_6
   - 774 p_up_0_6 <= 0
 non_overlap_any_0_7: p_left_0_7 + p_right_0_7 + p_down_0_7 + p_up_0_7
   >= 1
 non_overlap_left_0_7: w_var_0 + x_grid_var_0 - x_grid_var_7
   + 774 p_left_0_7 <= 774
 non_overlap_left_rev_0_7: - w_var_0 - x_grid_var_0 + x_grid_var_7
   - 774 p_left_0_7 <= 0
 non_overlap_right_0_7: w_var_7 - x_grid_var_0 + x_grid_var_7
   + 774 p_right_0_7 <= 774
 non_overlap_right_rev_0_7: - w_var_7 + x_grid_var_0 - x_grid_var_7
   - 774 p_right_0_7 <= 0
 non_overlap_down_0_7: h_var_0 + y_grid_var_0 - y_grid_var_7
   + 774 p_down_0_7 <= 774
 non_overlap_down_rev_0_7: - h_var_0 - y_grid_var_0 + y_grid_var_7
   - 774 p_down_0_7 <= 0
 non_overlap_up_0_7: h_var_7 - y_grid_var_0 + y_grid_var_7 + 774 p_up_0_7
   <= 774
 non_overlap_up_rev_0_7: - h_var_7 + y_grid_var_0 - y_grid_var_7
   - 774 p_up_0_7 <= 0
 non_overlap_any_0_8: p_left_0_8 + p_right_0_8 + p_down_0_8 + p_up_0_8
   >= 1
 non_overlap_left_0_8: w_var_0 + x_grid_var_0 - x_grid_var_8
   + 774 p_left_0_8 <= 774
 non_overlap_left_rev_0_8: - w_var_0 - x_grid_var_0 + x_grid_var_8
   - 774 p_left_0_8 <= 0
 non_overlap_right_0_8: w_var_8 - x_grid_var_0 + x_grid_var_8
   + 774 p_right_0_8 <= 774
 non_overlap_right_rev_0_8: - w_var_8 + x_grid_var_0 - x_grid_var_8
   - 774 p_right_0_8 <= 0
 non_overlap_down_0_8: h_var_0 + y_grid_var_0 - y_grid_var_8
   + 774 p_down_0_8 <= 774
 non_overlap_down_rev_0_8: - h_var_0 - y_grid_var_0 + y_grid_var_8
   - 774 p_down_0_8 <= 0
 non_overlap_up_0_8: h_var_8 - y_grid_var_0 + y_grid_var_8 + 774 p_up_0_8
   <= 774
 non_overlap_up_rev_0_8: - h_var_8 + y_grid_var_0 - y_grid_var_8
   - 774 p_up_0_8 <= 0
 non_overlap_any_0_9: p_left_0_9 + p_right_0_9 + p_down_0_9 + p_up_0_9
   >= 1
 non_overlap_left_0_9: w_var_0 + x_grid_var_0 - x_grid_var_9
   + 774 p_left_0_9 <= 774
 non_overlap_left_rev_0_9: - w_var_0 - x_grid_var_0 + x_grid_var_9
   - 774 p_left_0_9 <= 0
 non_overlap_right_0_9: w_var_9 - x_grid_var_0 + x_grid_var_9
   + 774 p_right_0_9 <= 774
 non_overlap_right_rev_0_9: - w_var_9 + x_grid_var_0 - x_grid_var_9
   - 774 p_right_0_9 <= 0
 non_overlap_down_0_9: h_var_0 + y_grid_var_0 - y_grid_var_9
   + 774 p_down_0_9 <= 774
 non_overlap_down_rev_0_9: - h_var_0 - y_grid_var_0 + y_grid_var_9
   - 774 p_down_0_9 <= 0
 non_overlap_up_0_9: h_var_9 - y_grid_var_0 + y_grid_var_9 + 774 p_up_0_9
   <= 774
 non_overlap_up_rev_0_9: - h_var_9 + y_grid_var_0 - y_grid_var_9
   - 774 p_up_0_9 <= 0
 non_overlap_any_0_10: p_left_0_10 + p_right_0_10 + p_down_0_10 + p_up_0_10
   >= 1
 non_overlap_left_0_10: w_var_0 + x_grid_var_0 - x_grid_var_10
   + 774 p_left_0_10 <= 774
 non_overlap_left_rev_0_10: - w_var_0 - x_grid_var_0 + x_grid_var_10
   - 774 p_left_0_10 <= 0
 non_overlap_right_0_10: w_var_10 - x_grid_var_0 + x_grid_var_10
   + 774 p_right_0_10 <= 774
 non_overlap_right_rev_0_10: - w_var_10 + x_grid_var_0 - x_grid_var_10
   - 774 p_right_0_10 <= 0
 non_overlap_down_0_10: h_var_0 + y_grid_var_0 - y_grid_var_10
   + 774 p_down_0_10 <= 774
 non_overlap_down_rev_0_10: - h_var_0 - y_grid_var_0 + y_grid_var_10
   - 774 p_down_0_10 <= 0
 non_overlap_up_0_10: h_var_10 - y_grid_var_0 + y_grid_var_10
   + 774 p_up_0_10 <= 774
 non_overlap_up_rev_0_10: - h_var_10 + y_grid_var_0 - y_grid_var_10
   - 774 p_up_0_10 <= 0
 non_overlap_any_0_11: p_left_0_11 + p_right_0_11 + p_down_0_11 + p_up_0_11
   >= 1
 non_overlap_left_0_11: w_var_0 + x_grid_var_0 - x_grid_var_11
   + 774 p_left_0_11 <= 774
 non_overlap_left_rev_0_11: - w_var_0 - x_grid_var_0 + x_grid_var_11
   - 774 p_left_0_11 <= 0
 non_overlap_right_0_11: w_var_11 - x_grid_var_0 + x_grid_var_11
   + 774 p_right_0_11 <= 774
 non_overlap_right_rev_0_11: - w_var_11 + x_grid_var_0 - x_grid_var_11
   - 774 p_right_0_11 <= 0
 non_overlap_down_0_11: h_var_0 + y_grid_var_0 - y_grid_var_11
   + 774 p_down_0_11 <= 774
 non_overlap_down_rev_0_11: - h_var_0 - y_grid_var_0 + y_grid_var_11
   - 774 p_down_0_11 <= 0
 non_overlap_up_0_11: h_var_11 - y_grid_var_0 + y_grid_var_11
   + 774 p_up_0_11 <= 774
 non_overlap_up_rev_0_11: - h_var_11 + y_grid_var_0 - y_grid_var_11
   - 774 p_up_0_11 <= 0
 non_overlap_any_0_12: p_left_0_12 + p_right_0_12 + p_down_0_12 + p_up_0_12
   >= 1
 non_overlap_left_0_12: w_var_0 + x_grid_var_0 - x_grid_var_12
   + 774 p_left_0_12 <= 774
 non_overlap_left_rev_0_12: - w_var_0 - x_grid_var_0 + x_grid_var_12
   - 774 p_left_0_12 <= 0
 non_overlap_right_0_12: w_var_12 - x_grid_var_0 + x_grid_var_12
   + 774 p_right_0_12 <= 774
 non_overlap_right_rev_0_12: - w_var_12 + x_grid_var_0 - x_grid_var_12
   - 774 p_right_0_12 <= 0
 non_overlap_down_0_12: h_var_0 + y_grid_var_0 - y_grid_var_12
   + 774 p_down_0_12 <= 774
 non_overlap_down_rev_0_12: - h_var_0 - y_grid_var_0 + y_grid_var_12
   - 774 p_down_0_12 <= 0
 non_overlap_up_0_12: h_var_12 - y_grid_var_0 + y_grid_var_12
   + 774 p_up_0_12 <= 774
 non_overlap_up_rev_0_12: - h_var_12 + y_grid_var_0 - y_grid_var_12
   - 774 p_up_0_12 <= 0
 non_overlap_any_0_13: p_left_0_13 + p_right_0_13 + p_down_0_13 + p_up_0_13
   >= 1
 non_overlap_left_0_13: w_var_0 + x_grid_var_0 - x_grid_var_13
   + 774 p_left_0_13 <= 774
 non_overlap_left_rev_0_13: - w_var_0 - x_grid_var_0 + x_grid_var_13
   - 774 p_left_0_13 <= 0
 non_overlap_right_0_13: w_var_13 - x_grid_var_0 + x_grid_var_13
   + 774 p_right_0_13 <= 774
 non_overlap_right_rev_0_13: - w_var_13 + x_grid_var_0 - x_grid_var_13
   - 774 p_right_0_13 <= 0
 non_overlap_down_0_13: h_var_0 + y_grid_var_0 - y_grid_var_13
   + 774 p_down_0_13 <= 774
 non_overlap_down_rev_0_13: - h_var_0 - y_grid_var_0 + y_grid_var_13
   - 774 p_down_0_13 <= 0
 non_overlap_up_0_13: h_var_13 - y_grid_var_0 + y_grid_var_13
   + 774 p_up_0_13 <= 774
 non_overlap_up_rev_0_13: - h_var_13 + y_grid_var_0 - y_grid_var_13
   - 774 p_up_0_13 <= 0
 non_overlap_any_0_14: p_left_0_14 + p_right_0_14 + p_down_0_14 + p_up_0_14
   >= 1
 non_overlap_left_0_14: w_var_0 + x_grid_var_0 - x_grid_var_14
   + 774 p_left_0_14 <= 774
 non_overlap_left_rev_0_14: - w_var_0 - x_grid_var_0 + x_grid_var_14
   - 774 p_left_0_14 <= 0
 non_overlap_right_0_14: w_var_14 - x_grid_var_0 + x_grid_var_14
   + 774 p_right_0_14 <= 774
 non_overlap_right_rev_0_14: - w_var_14 + x_grid_var_0 - x_grid_var_14
   - 774 p_right_0_14 <= 0
 non_overlap_down_0_14: h_var_0 + y_grid_var_0 - y_grid_var_14
   + 774 p_down_0_14 <= 774
 non_overlap_down_rev_0_14: - h_var_0 - y_grid_var_0 + y_grid_var_14
   - 774 p_down_0_14 <= 0
 non_overlap_up_0_14: h_var_14 - y_grid_var_0 + y_grid_var_14
   + 774 p_up_0_14 <= 774
 non_overlap_up_rev_0_14: - h_var_14 + y_grid_var_0 - y_grid_var_14
   - 774 p_up_0_14 <= 0
 non_overlap_any_0_15: p_left_0_15 + p_right_0_15 + p_down_0_15 + p_up_0_15
   >= 1
 non_overlap_left_0_15: w_var_0 + x_grid_var_0 - x_grid_var_15
   + 774 p_left_0_15 <= 774
 non_overlap_left_rev_0_15: - w_var_0 - x_grid_var_0 + x_grid_var_15
   - 774 p_left_0_15 <= 0
 non_overlap_right_0_15: w_var_15 - x_grid_var_0 + x_grid_var_15
   + 774 p_right_0_15 <= 774
 non_overlap_right_rev_0_15: - w_var_15 + x_grid_var_0 - x_grid_var_15
   - 774 p_right_0_15 <= 0
 non_overlap_down_0_15: h_var_0 + y_grid_var_0 - y_grid_var_15
   + 774 p_down_0_15 <= 774
 non_overlap_down_rev_0_15: - h_var_0 - y_grid_var_0 + y_grid_var_15
   - 774 p_down_0_15 <= 0
 non_overlap_up_0_15: h_var_15 - y_grid_var_0 + y_grid_var_15
   + 774 p_up_0_15 <= 774
 non_overlap_up_rev_0_15: - h_var_15 + y_grid_var_0 - y_grid_var_15
   - 774 p_up_0_15 <= 0
 non_overlap_any_1_2: p_left_1_2 + p_right_1_2 + p_down_1_2 + p_up_1_2
   >= 1
 non_overlap_left_1_2: w_var_1 + x_grid_var_1 - x_grid_var_2
   + 774 p_left_1_2 <= 774
 non_overlap_left_rev_1_2: - w_var_1 - x_grid_var_1 + x_grid_var_2
   - 774 p_left_1_2 <= 0
 non_overlap_right_1_2: w_var_2 - x_grid_var_1 + x_grid_var_2
   + 774 p_right_1_2 <= 774
 non_overlap_right_rev_1_2: - w_var_2 + x_grid_var_1 - x_grid_var_2
   - 774 p_right_1_2 <= 0
 non_overlap_down_1_2: h_var_1 + y_grid_var_1 - y_grid_var_2
   + 774 p_down_1_2 <= 774
 non_overlap_down_rev_1_2: - h_var_1 - y_grid_var_1 + y_grid_var_2
   - 774 p_down_1_2 <= 0
 non_overlap_up_1_2: h_var_2 - y_grid_var_1 + y_grid_var_2 + 774 p_up_1_2
   <= 774
 non_overlap_up_rev_1_2: - h_var_2 + y_grid_var_1 - y_grid_var_2
   - 774 p_up_1_2 <= 0
 non_overlap_any_1_3: p_left_1_3 + p_right_1_3 + p_down_1_3 + p_up_1_3
   >= 1
 non_overlap_left_1_3: w_var_1 + x_grid_var_1 - x_grid_var_3
   + 774 p_left_1_3 <= 774
 non_overlap_left_rev_1_3: - w_var_1 - x_grid_var_1 + x_grid_var_3
   - 774 p_left_1_3 <= 0
 non_overlap_right_1_3: w_var_3 - x_grid_var_1 + x_grid_var_3
   + 774 p_right_1_3 <= 774
 non_overlap_right_rev_1_3: - w_var_3 + x_grid_var_1 - x_grid_var_3
   - 774 p_right_1_3 <= 0
 non_overlap_down_1_3: h_var_1 + y_grid_var_1 - y_grid_var_3
   + 774 p_down_1_3 <= 774
 non_overlap_down_rev_1_3: - h_var_1 - y_grid_var_1 + y_grid_var_3
   - 774 p_down_1_3 <= 0
 non_overlap_up_1_3: h_var_3 - y_grid_var_1 + y_grid_var_3 + 774 p_up_1_3
   <= 774
 non_overlap_up_rev_1_3: - h_var_3 + y_grid_var_1 - y_grid_var_3
   - 774 p_up_1_3 <= 0
 non_overlap_any_1_4: p_left_1_4 + p_right_1_4 + p_down_1_4 + p_up_1_4
   >= 1
 non_overlap_left_1_4: w_var_1 + x_grid_var_1 - x_grid_var_4
   + 774 p_left_1_4 <= 774
 non_overlap_left_rev_1_4: - w_var_1 - x_grid_var_1 + x_grid_var_4
   - 774 p_left_1_4 <= 0
 non_overlap_right_1_4: w_var_4 - x_grid_var_1 + x_grid_var_4
   + 774 p_right_1_4 <= 774
 non_overlap_right_rev_1_4: - w_var_4 + x_grid_var_1 - x_grid_var_4
   - 774 p_right_1_4 <= 0
 non_overlap_down_1_4: h_var_1 + y_grid_var_1 - y_grid_var_4
   + 774 p_down_1_4 <= 774
 non_overlap_down_rev_1_4: - h_var_1 - y_grid_var_1 + y_grid_var_4
   - 774 p_down_1_4 <= 0
 non_overlap_up_1_4: h_var_4 - y_grid_var_1 + y_grid_var_4 + 774 p_up_1_4
   <= 774
 non_overlap_up_rev_1_4: - h_var_4 + y_grid_var_1 - y_grid_var_4
   - 774 p_up_1_4 <= 0
 non_overlap_any_1_5: p_left_1_5 + p_right_1_5 + p_down_1_5 + p_up_1_5
   >= 1
 non_overlap_left_1_5: w_var_1 + x_grid_var_1 - x_grid_var_5
   + 774 p_left_1_5 <= 774
 non_overlap_left_rev_1_5: - w_var_1 - x_grid_var_1 + x_grid_var_5
   - 774 p_left_1_5 <= 0
 non_overlap_right_1_5: w_var_5 - x_grid_var_1 + x_grid_var_5
   + 774 p_right_1_5 <= 774
 non_overlap_right_rev_1_5: - w_var_5 + x_grid_var_1 - x_grid_var_5
   - 774 p_right_1_5 <= 0
 non_overlap_down_1_5: h_var_1 + y_grid_var_1 - y_grid_var_5
   + 774 p_down_1_5 <= 774
 non_overlap_down_rev_1_5: - h_var_1 - y_grid_var_1 + y_grid_var_5
   - 774 p_down_1_5 <= 0
 non_overlap_up_1_5: h_var_5 - y_grid_var_1 + y_grid_var_5 + 774 p_up_1_5
   <= 774
 non_overlap_up_rev_1_5: - h_var_5 + y_grid_var_1 - y_grid_var_5
   - 774 p_up_1_5 <= 0
 non_overlap_any_1_6: p_left_1_6 + p_right_1_6 + p_down_1_6 + p_up_1_6
   >= 1
 non_overlap_left_1_6: w_var_1 + x_grid_var_1 - x_grid_var_6
   + 774 p_left_1_6 <= 774
 non_overlap_left_rev_1_6: - w_var_1 - x_grid_var_1 + x_grid_var_6
   - 774 p_left_1_6 <= 0
 non_overlap_right_1_6: w_var_6 - x_grid_var_1 + x_grid_var_6
   + 774 p_right_1_6 <= 774
 non_overlap_right_rev_1_6: - w_var_6 + x_grid_var_1 - x_grid_var_6
   - 774 p_right_1_6 <= 0
 non_overlap_down_1_6: h_var_1 + y_grid_var_1 - y_grid_var_6
   + 774 p_down_1_6 <= 774
 non_overlap_down_rev_1_6: - h_var_1 - y_grid_var_1 + y_grid_var_6
   - 774 p_down_1_6 <= 0
 non_overlap_up_1_6: h_var_6 - y_grid_var_1 + y_grid_var_6 + 774 p_up_1_6
   <= 774
 non_overlap_up_rev_1_6: - h_var_6 + y_grid_var_1 - y_grid_var_6
   - 774 p_up_1_6 <= 0
 non_overlap_any_1_7: p_left_1_7 + p_right_1_7 + p_down_1_7 + p_up_1_7
   >= 1
 non_overlap_left_1_7: w_var_1 + x_grid_var_1 - x_grid_var_7
   + 774 p_left_1_7 <= 774
 non_overlap_left_rev_1_7: - w_var_1 - x_grid_var_1 + x_grid_var_7
   - 774 p_left_1_7 <= 0
 non_overlap_right_1_7: w_var_7 - x_grid_var_1 + x_grid_var_7
   + 774 p_right_1_7 <= 774
 non_overlap_right_rev_1_7: - w_var_7 + x_grid_var_1 - x_grid_var_7
   - 774 p_right_1_7 <= 0
 non_overlap_down_1_7: h_var_1 + y_grid_var_1 - y_grid_var_7
   + 774 p_down_1_7 <= 774
 non_overlap_down_rev_1_7: - h_var_1 - y_grid_var_1 + y_grid_var_7
   - 774 p_down_1_7 <= 0
 non_overlap_up_1_7: h_var_7 - y_grid_var_1 + y_grid_var_7 + 774 p_up_1_7
   <= 774
 non_overlap_up_rev_1_7: - h_var_7 + y_grid_var_1 - y_grid_var_7
   - 774 p_up_1_7 <= 0
 non_overlap_any_1_8: p_left_1_8 + p_right_1_8 + p_down_1_8 + p_up_1_8
   >= 1
 non_overlap_left_1_8: w_var_1 + x_grid_var_1 - x_grid_var_8
   + 774 p_left_1_8 <= 774
 non_overlap_left_rev_1_8: - w_var_1 - x_grid_var_1 + x_grid_var_8
   - 774 p_left_1_8 <= 0
 non_overlap_right_1_8: w_var_8 - x_grid_var_1 + x_grid_var_8
   + 774 p_right_1_8 <= 774
 non_overlap_right_rev_1_8: - w_var_8 + x_grid_var_1 - x_grid_var_8
   - 774 p_right_1_8 <= 0
 non_overlap_down_1_8: h_var_1 + y_grid_var_1 - y_grid_var_8
   + 774 p_down_1_8 <= 774
 non_overlap_down_rev_1_8: - h_var_1 - y_grid_var_1 + y_grid_var_8
   - 774 p_down_1_8 <= 0
 non_overlap_up_1_8: h_var_8 - y_grid_var_1 + y_grid_var_8 + 774 p_up_1_8
   <= 774
 non_overlap_up_rev_1_8: - h_var_8 + y_grid_var_1 - y_grid_var_8
   - 774 p_up_1_8 <= 0
 non_overlap_any_1_9: p_left_1_9 + p_right_1_9 + p_down_1_9 + p_up_1_9
   >= 1
 non_overlap_left_1_9: w_var_1 + x_grid_var_1 - x_grid_var_9
   + 774 p_left_1_9 <= 774
 non_overlap_left_rev_1_9: - w_var_1 - x_grid_var_1 + x_grid_var_9
   - 774 p_left_1_9 <= 0
 non_overlap_right_1_9: w_var_9 - x_grid_var_1 + x_grid_var_9
   + 774 p_right_1_9 <= 774
 non_overlap_right_rev_1_9: - w_var_9 + x_grid_var_1 - x_grid_var_9
   - 774 p_right_1_9 <= 0
 non_overlap_down_1_9: h_var_1 + y_grid_var_1 - y_grid_var_9
   + 774 p_down_1_9 <= 774
 non_overlap_down_rev_1_9: - h_var_1 - y_grid_var_1 + y_grid_var_9
   - 774 p_down_1_9 <= 0
 non_overlap_up_1_9: h_var_9 - y_grid_var_1 + y_grid_var_9 + 774 p_up_1_9
   <= 774
 non_overlap_up_rev_1_9: - h_var_9 + y_grid_var_1 - y_grid_var_9
   - 774 p_up_1_9 <= 0
 non_overlap_any_1_10: p_left_1_10 + p_right_1_10 + p_down_1_10 + p_up_1_10
   >= 1
 non_overlap_left_1_10: w_var_1 + x_grid_var_1 - x_grid_var_10
   + 774 p_left_1_10 <= 774
 non_overlap_left_rev_1_10: - w_var_1 - x_grid_var_1 + x_grid_var_10
   - 774 p_left_1_10 <= 0
 non_overlap_right_1_10: w_var_10 - x_grid_var_1 + x_grid_var_10
   + 774 p_right_1_10 <= 774
 non_overlap_right_rev_1_10: - w_var_10 + x_grid_var_1 - x_grid_var_10
   - 774 p_right_1_10 <= 0
 non_overlap_down_1_10: h_var_1 + y_grid_var_1 - y_grid_var_10
   + 774 p_down_1_10 <= 774
 non_overlap_down_rev_1_10: - h_var_1 - y_grid_var_1 + y_grid_var_10
   - 774 p_down_1_10 <= 0
 non_overlap_up_1_10: h_var_10 - y_grid_var_1 + y_grid_var_10
   + 774 p_up_1_10 <= 774
 non_overlap_up_rev_1_10: - h_var_10 + y_grid_var_1 - y_grid_var_10
   - 774 p_up_1_10 <= 0
 non_overlap_any_1_11: p_left_1_11 + p_right_1_11 + p_down_1_11 + p_up_1_11
   >= 1
 non_overlap_left_1_11: w_var_1 + x_grid_var_1 - x_grid_var_11
   + 774 p_left_1_11 <= 774
 non_overlap_left_rev_1_11: - w_var_1 - x_grid_var_1 + x_grid_var_11
   - 774 p_left_1_11 <= 0
 non_overlap_right_1_11: w_var_11 - x_grid_var_1 + x_grid_var_11
   + 774 p_right_1_11 <= 774
 non_overlap_right_rev_1_11: - w_var_11 + x_grid_var_1 - x_grid_var_11
   - 774 p_right_1_11 <= 0
 non_overlap_down_1_11: h_var_1 + y_grid_var_1 - y_grid_var_11
   + 774 p_down_1_11 <= 774
 non_overlap_down_rev_1_11: - h_var_1 - y_grid_var_1 + y_grid_var_11
   - 774 p_down_1_11 <= 0
 non_overlap_up_1_11: h_var_11 - y_grid_var_1 + y_grid_var_11
   + 774 p_up_1_11 <= 774
 non_overlap_up_rev_1_11: - h_var_11 + y_grid_var_1 - y_grid_var_11
   - 774 p_up_1_11 <= 0
 non_overlap_any_1_12: p_left_1_12 + p_right_1_12 + p_down_1_12 + p_up_1_12
   >= 1
 non_overlap_left_1_12: w_var_1 + x_grid_var_1 - x_grid_var_12
   + 774 p_left_1_12 <= 774
 non_overlap_left_rev_1_12: - w_var_1 - x_grid_var_1 + x_grid_var_12
   - 774 p_left_1_12 <= 0
 non_overlap_right_1_12: w_var_12 - x_grid_var_1 + x_grid_var_12
   + 774 p_right_1_12 <= 774
 non_overlap_right_rev_1_12: - w_var_12 + x_grid_var_1 - x_grid_var_12
   - 774 p_right_1_12 <= 0
 non_overlap_down_1_12: h_var_1 + y_grid_var_1 - y_grid_var_12
   + 774 p_down_1_12 <= 774
 non_overlap_down_rev_1_12: - h_var_1 - y_grid_var_1 + y_grid_var_12
   - 774 p_down_1_12 <= 0
 non_overlap_up_1_12: h_var_12 - y_grid_var_1 + y_grid_var_12
   + 774 p_up_1_12 <= 774
 non_overlap_up_rev_1_12: - h_var_12 + y_grid_var_1 - y_grid_var_12
   - 774 p_up_1_12 <= 0
 non_overlap_any_1_13: p_left_1_13 + p_right_1_13 + p_down_1_13 + p_up_1_13
   >= 1
 non_overlap_left_1_13: w_var_1 + x_grid_var_1 - x_grid_var_13
   + 774 p_left_1_13 <= 774
 non_overlap_left_rev_1_13: - w_var_1 - x_grid_var_1 + x_grid_var_13
   - 774 p_left_1_13 <= 0
 non_overlap_right_1_13: w_var_13 - x_grid_var_1 + x_grid_var_13
   + 774 p_right_1_13 <= 774
 non_overlap_right_rev_1_13: - w_var_13 + x_grid_var_1 - x_grid_var_13
   - 774 p_right_1_13 <= 0
 non_overlap_down_1_13: h_var_1 + y_grid_var_1 - y_grid_var_13
   + 774 p_down_1_13 <= 774
 non_overlap_down_rev_1_13: - h_var_1 - y_grid_var_1 + y_grid_var_13
   - 774 p_down_1_13 <= 0
 non_overlap_up_1_13: h_var_13 - y_grid_var_1 + y_grid_var_13
   + 774 p_up_1_13 <= 774
 non_overlap_up_rev_1_13: - h_var_13 + y_grid_var_1 - y_grid_var_13
   - 774 p_up_1_13 <= 0
 non_overlap_any_1_14: p_left_1_14 + p_right_1_14 + p_down_1_14 + p_up_1_14
   >= 1
 non_overlap_left_1_14: w_var_1 + x_grid_var_1 - x_grid_var_14
   + 774 p_left_1_14 <= 774
 non_overlap_left_rev_1_14: - w_var_1 - x_grid_var_1 + x_grid_var_14
   - 774 p_left_1_14 <= 0
 non_overlap_right_1_14: w_var_14 - x_grid_var_1 + x_grid_var_14
   + 774 p_right_1_14 <= 774
 non_overlap_right_rev_1_14: - w_var_14 + x_grid_var_1 - x_grid_var_14
   - 774 p_right_1_14 <= 0
 non_overlap_down_1_14: h_var_1 + y_grid_var_1 - y_grid_var_14
   + 774 p_down_1_14 <= 774
 non_overlap_down_rev_1_14: - h_var_1 - y_grid_var_1 + y_grid_var_14
   - 774 p_down_1_14 <= 0
 non_overlap_up_1_14: h_var_14 - y_grid_var_1 + y_grid_var_14
   + 774 p_up_1_14 <= 774
 non_overlap_up_rev_1_14: - h_var_14 + y_grid_var_1 - y_grid_var_14
   - 774 p_up_1_14 <= 0
 non_overlap_any_1_15: p_left_1_15 + p_right_1_15 + p_down_1_15 + p_up_1_15
   >= 1
 non_overlap_left_1_15: w_var_1 + x_grid_var_1 - x_grid_var_15
   + 774 p_left_1_15 <= 774
 non_overlap_left_rev_1_15: - w_var_1 - x_grid_var_1 + x_grid_var_15
   - 774 p_left_1_15 <= 0
 non_overlap_right_1_15: w_var_15 - x_grid_var_1 + x_grid_var_15
   + 774 p_right_1_15 <= 774
 non_overlap_right_rev_1_15: - w_var_15 + x_grid_var_1 - x_grid_var_15
   - 774 p_right_1_15 <= 0
 non_overlap_down_1_15: h_var_1 + y_grid_var_1 - y_grid_var_15
   + 774 p_down_1_15 <= 774
 non_overlap_down_rev_1_15: - h_var_1 - y_grid_var_1 + y_grid_var_15
   - 774 p_down_1_15 <= 0
 non_overlap_up_1_15: h_var_15 - y_grid_var_1 + y_grid_var_15
   + 774 p_up_1_15 <= 774
 non_overlap_up_rev_1_15: - h_var_15 + y_grid_var_1 - y_grid_var_15
   - 774 p_up_1_15 <= 0
 non_overlap_any_2_3: p_left_2_3 + p_right_2_3 + p_down_2_3 + p_up_2_3
   >= 1
 non_overlap_left_2_3: w_var_2 + x_grid_var_2 - x_grid_var_3
   + 774 p_left_2_3 <= 774
 non_overlap_left_rev_2_3: - w_var_2 - x_grid_var_2 + x_grid_var_3
   - 774 p_left_2_3 <= 0
 non_overlap_right_2_3: w_var_3 - x_grid_var_2 + x_grid_var_3
   + 774 p_right_2_3 <= 774
 non_overlap_right_rev_2_3: - w_var_3 + x_grid_var_2 - x_grid_var_3
   - 774 p_right_2_3 <= 0
 non_overlap_down_2_3: h_var_2 + y_grid_var_2 - y_grid_var_3
   + 774 p_down_2_3 <= 774
 non_overlap_down_rev_2_3: - h_var_2 - y_grid_var_2 + y_grid_var_3
   - 774 p_down_2_3 <= 0
 non_overlap_up_2_3: h_var_3 - y_grid_var_2 + y_grid_var_3 + 774 p_up_2_3
   <= 774
 non_overlap_up_rev_2_3: - h_var_3 + y_grid_var_2 - y_grid_var_3
   - 774 p_up_2_3 <= 0
 non_overlap_any_2_4: p_left_2_4 + p_right_2_4 + p_down_2_4 + p_up_2_4
   >= 1
 non_overlap_left_2_4: w_var_2 + x_grid_var_2 - x_grid_var_4
   + 774 p_left_2_4 <= 774
 non_overlap_left_rev_2_4: - w_var_2 - x_grid_var_2 + x_grid_var_4
   - 774 p_left_2_4 <= 0
 non_overlap_right_2_4: w_var_4 - x_grid_var_2 + x_grid_var_4
   + 774 p_right_2_4 <= 774
 non_overlap_right_rev_2_4: - w_var_4 + x_grid_var_2 - x_grid_var_4
   - 774 p_right_2_4 <= 0
 non_overlap_down_2_4: h_var_2 + y_grid_var_2 - y_grid_var_4
   + 774 p_down_2_4 <= 774
 non_overlap_down_rev_2_4: - h_var_2 - y_grid_var_2 + y_grid_var_4
   - 774 p_down_2_4 <= 0
 non_overlap_up_2_4: h_var_4 - y_grid_var_2 + y_grid_var_4 + 774 p_up_2_4
   <= 774
 non_overlap_up_rev_2_4: - h_var_4 + y_grid_var_2 - y_grid_var_4
   - 774 p_up_2_4 <= 0
 non_overlap_any_2_5: p_left_2_5 + p_right_2_5 + p_down_2_5 + p_up_2_5
   >= 1
 non_overlap_left_2_5: w_var_2 + x_grid_var_2 - x_grid_var_5
   + 774 p_left_2_5 <= 774
 non_overlap_left_rev_2_5: - w_var_2 - x_grid_var_2 + x_grid_var_5
   - 774 p_left_2_5 <= 0
 non_overlap_right_2_5: w_var_5 - x_grid_var_2 + x_grid_var_5
   + 774 p_right_2_5 <= 774
 non_overlap_right_rev_2_5: - w_var_5 + x_grid_var_2 - x_grid_var_5
   - 774 p_right_2_5 <= 0
 non_overlap_down_2_5: h_var_2 + y_grid_var_2 - y_grid_var_5
   + 774 p_down_2_5 <= 774
 non_overlap_down_rev_2_5: - h_var_2 - y_grid_var_2 + y_grid_var_5
   - 774 p_down_2_5 <= 0
 non_overlap_up_2_5: h_var_5 - y_grid_var_2 + y_grid_var_5 + 774 p_up_2_5
   <= 774
 non_overlap_up_rev_2_5: - h_var_5 + y_grid_var_2 - y_grid_var_5
   - 774 p_up_2_5 <= 0
 non_overlap_any_2_6: p_left_2_6 + p_right_2_6 + p_down_2_6 + p_up_2_6
   >= 1
 non_overlap_left_2_6: w_var_2 + x_grid_var_2 - x_grid_var_6
   + 774 p_left_2_6 <= 774
 non_overlap_left_rev_2_6: - w_var_2 - x_grid_var_2 + x_grid_var_6
   - 774 p_left_2_6 <= 0
 non_overlap_right_2_6: w_var_6 - x_grid_var_2 + x_grid_var_6
   + 774 p_right_2_6 <= 774
 non_overlap_right_rev_2_6: - w_var_6 + x_grid_var_2 - x_grid_var_6
   - 774 p_right_2_6 <= 0
 non_overlap_down_2_6: h_var_2 + y_grid_var_2 - y_grid_var_6
   + 774 p_down_2_6 <= 774
 non_overlap_down_rev_2_6: - h_var_2 - y_grid_var_2 + y_grid_var_6
   - 774 p_down_2_6 <= 0
 non_overlap_up_2_6: h_var_6 - y_grid_var_2 + y_grid_var_6 + 774 p_up_2_6
   <= 774
 non_overlap_up_rev_2_6: - h_var_6 + y_grid_var_2 - y_grid_var_6
   - 774 p_up_2_6 <= 0
 non_overlap_any_2_7: p_left_2_7 + p_right_2_7 + p_down_2_7 + p_up_2_7
   >= 1
 non_overlap_left_2_7: w_var_2 + x_grid_var_2 - x_grid_var_7
   + 774 p_left_2_7 <= 774
 non_overlap_left_rev_2_7: - w_var_2 - x_grid_var_2 + x_grid_var_7
   - 774 p_left_2_7 <= 0
 non_overlap_right_2_7: w_var_7 - x_grid_var_2 + x_grid_var_7
   + 774 p_right_2_7 <= 774
 non_overlap_right_rev_2_7: - w_var_7 + x_grid_var_2 - x_grid_var_7
   - 774 p_right_2_7 <= 0
 non_overlap_down_2_7: h_var_2 + y_grid_var_2 - y_grid_var_7
   + 774 p_down_2_7 <= 774
 non_overlap_down_rev_2_7: - h_var_2 - y_grid_var_2 + y_grid_var_7
   - 774 p_down_2_7 <= 0
 non_overlap_up_2_7: h_var_7 - y_grid_var_2 + y_grid_var_7 + 774 p_up_2_7
   <= 774
 non_overlap_up_rev_2_7: - h_var_7 + y_grid_var_2 - y_grid_var_7
   - 774 p_up_2_7 <= 0
 non_overlap_any_2_8: p_left_2_8 + p_right_2_8 + p_down_2_8 + p_up_2_8
   >= 1
 non_overlap_left_2_8: w_var_2 + x_grid_var_2 - x_grid_var_8
   + 774 p_left_2_8 <= 774
 non_overlap_left_rev_2_8: - w_var_2 - x_grid_var_2 + x_grid_var_8
   - 774 p_left_2_8 <= 0
 non_overlap_right_2_8: w_var_8 - x_grid_var_2 + x_grid_var_8
   + 774 p_right_2_8 <= 774
 non_overlap_right_rev_2_8: - w_var_8 + x_grid_var_2 - x_grid_var_8
   - 774 p_right_2_8 <= 0
 non_overlap_down_2_8: h_var_2 + y_grid_var_2 - y_grid_var_8
   + 774 p_down_2_8 <= 774
 non_overlap_down_rev_2_8: - h_var_2 - y_grid_var_2 + y_grid_var_8
   - 774 p_down_2_8 <= 0
 non_overlap_up_2_8: h_var_8 - y_grid_var_2 + y_grid_var_8 + 774 p_up_2_8
   <= 774
 non_overlap_up_rev_2_8: - h_var_8 + y_grid_var_2 - y_grid_var_8
   - 774 p_up_2_8 <= 0
 non_overlap_any_2_9: p_left_2_9 + p_right_2_9 + p_down_2_9 + p_up_2_9
   >= 1
 non_overlap_left_2_9: w_var_2 + x_grid_var_2 - x_grid_var_9
   + 774 p_left_2_9 <= 774
 non_overlap_left_rev_2_9: - w_var_2 - x_grid_var_2 + x_grid_var_9
   - 774 p_left_2_9 <= 0
 non_overlap_right_2_9: w_var_9 - x_grid_var_2 + x_grid_var_9
   + 774 p_right_2_9 <= 774
 non_overlap_right_rev_2_9: - w_var_9 + x_grid_var_2 - x_grid_var_9
   - 774 p_right_2_9 <= 0
 non_overlap_down_2_9: h_var_2 + y_grid_var_2 - y_grid_var_9
   + 774 p_down_2_9 <= 774
 non_overlap_down_rev_2_9: - h_var_2 - y_grid_var_2 + y_grid_var_9
   - 774 p_down_2_9 <= 0
 non_overlap_up_2_9: h_var_9 - y_grid_var_2 + y_grid_var_9 + 774 p_up_2_9
   <= 774
 non_overlap_up_rev_2_9: - h_var_9 + y_grid_var_2 - y_grid_var_9
   - 774 p_up_2_9 <= 0
 non_overlap_any_2_10: p_left_2_10 + p_right_2_10 + p_down_2_10 + p_up_2_10
   >= 1
 non_overlap_left_2_10: w_var_2 + x_grid_var_2 - x_grid_var_10
   + 774 p_left_2_10 <= 774
 non_overlap_left_rev_2_10: - w_var_2 - x_grid_var_2 + x_grid_var_10
   - 774 p_left_2_10 <= 0
 non_overlap_right_2_10: w_var_10 - x_grid_var_2 + x_grid_var_10
   + 774 p_right_2_10 <= 774
 non_overlap_right_rev_2_10: - w_var_10 + x_grid_var_2 - x_grid_var_10
   - 774 p_right_2_10 <= 0
 non_overlap_down_2_10: h_var_2 + y_grid_var_2 - y_grid_var_10
   + 774 p_down_2_10 <= 774
 non_overlap_down_rev_2_10: - h_var_2 - y_grid_var_2 + y_grid_var_10
   - 774 p_down_2_10 <= 0
 non_overlap_up_2_10: h_var_10 - y_grid_var_2 + y_grid_var_10
   + 774 p_up_2_10 <= 774
 non_overlap_up_rev_2_10: - h_var_10 + y_grid_var_2 - y_grid_var_10
   - 774 p_up_2_10 <= 0
 non_overlap_any_2_11: p_left_2_11 + p_right_2_11 + p_down_2_11 + p_up_2_11
   >= 1
 non_overlap_left_2_11: w_var_2 + x_grid_var_2 - x_grid_var_11
   + 774 p_left_2_11 <= 774
 non_overlap_left_rev_2_11: - w_var_2 - x_grid_var_2 + x_grid_var_11
   - 774 p_left_2_11 <= 0
 non_overlap_right_2_11: w_var_11 - x_grid_var_2 + x_grid_var_11
   + 774 p_right_2_11 <= 774
 non_overlap_right_rev_2_11: - w_var_11 + x_grid_var_2 - x_grid_var_11
   - 774 p_right_2_11 <= 0
 non_overlap_down_2_11: h_var_2 + y_grid_var_2 - y_grid_var_11
   + 774 p_down_2_11 <= 774
 non_overlap_down_rev_2_11: - h_var_2 - y_grid_var_2 + y_grid_var_11
   - 774 p_down_2_11 <= 0
 non_overlap_up_2_11: h_var_11 - y_grid_var_2 + y_grid_var_11
   + 774 p_up_2_11 <= 774
 non_overlap_up_rev_2_11: - h_var_11 + y_grid_var_2 - y_grid_var_11
   - 774 p_up_2_11 <= 0
 non_overlap_any_2_12: p_left_2_12 + p_right_2_12 + p_down_2_12 + p_up_2_12
   >= 1
 non_overlap_left_2_12: w_var_2 + x_grid_var_2 - x_grid_var_12
   + 774 p_left_2_12 <= 774
 non_overlap_left_rev_2_12: - w_var_2 - x_grid_var_2 + x_grid_var_12
   - 774 p_left_2_12 <= 0
 non_overlap_right_2_12: w_var_12 - x_grid_var_2 + x_grid_var_12
   + 774 p_right_2_12 <= 774
 non_overlap_right_rev_2_12: - w_var_12 + x_grid_var_2 - x_grid_var_12
   - 774 p_right_2_12 <= 0
 non_overlap_down_2_12: h_var_2 + y_grid_var_2 - y_grid_var_12
   + 774 p_down_2_12 <= 774
 non_overlap_down_rev_2_12: - h_var_2 - y_grid_var_2 + y_grid_var_12
   - 774 p_down_2_12 <= 0
 non_overlap_up_2_12: h_var_12 - y_grid_var_2 + y_grid_var_12
   + 774 p_up_2_12 <= 774
 non_overlap_up_rev_2_12: - h_var_12 + y_grid_var_2 - y_grid_var_12
   - 774 p_up_2_12 <= 0
 non_overlap_any_2_13: p_left_2_13 + p_right_2_13 + p_down_2_13 + p_up_2_13
   >= 1
 non_overlap_left_2_13: w_var_2 + x_grid_var_2 - x_grid_var_13
   + 774 p_left_2_13 <= 774
 non_overlap_left_rev_2_13: - w_var_2 - x_grid_var_2 + x_grid_var_13
   - 774 p_left_2_13 <= 0
 non_overlap_right_2_13: w_var_13 - x_grid_var_2 + x_grid_var_13
   + 774 p_right_2_13 <= 774
 non_overlap_right_rev_2_13: - w_var_13 + x_grid_var_2 - x_grid_var_13
   - 774 p_right_2_13 <= 0
 non_overlap_down_2_13: h_var_2 + y_grid_var_2 - y_grid_var_13
   + 774 p_down_2_13 <= 774
 non_overlap_down_rev_2_13: - h_var_2 - y_grid_var_2 + y_grid_var_13
   - 774 p_down_2_13 <= 0
 non_overlap_up_2_13: h_var_13 - y_grid_var_2 + y_grid_var_13
   + 774 p_up_2_13 <= 774
 non_overlap_up_rev_2_13: - h_var_13 + y_grid_var_2 - y_grid_var_13
   - 774 p_up_2_13 <= 0
 non_overlap_any_2_14: p_left_2_14 + p_right_2_14 + p_down_2_14 + p_up_2_14
   >= 1
 non_overlap_left_2_14: w_var_2 + x_grid_var_2 - x_grid_var_14
   + 774 p_left_2_14 <= 774
 non_overlap_left_rev_2_14: - w_var_2 - x_grid_var_2 + x_grid_var_14
   - 774 p_left_2_14 <= 0
 non_overlap_right_2_14: w_var_14 - x_grid_var_2 + x_grid_var_14
   + 774 p_right_2_14 <= 774
 non_overlap_right_rev_2_14: - w_var_14 + x_grid_var_2 - x_grid_var_14
   - 774 p_right_2_14 <= 0
 non_overlap_down_2_14: h_var_2 + y_grid_var_2 - y_grid_var_14
   + 774 p_down_2_14 <= 774
 non_overlap_down_rev_2_14: - h_var_2 - y_grid_var_2 + y_grid_var_14
   - 774 p_down_2_14 <= 0
 non_overlap_up_2_14: h_var_14 - y_grid_var_2 + y_grid_var_14
   + 774 p_up_2_14 <= 774
 non_overlap_up_rev_2_14: - h_var_14 + y_grid_var_2 - y_grid_var_14
   - 774 p_up_2_14 <= 0
 non_overlap_any_2_15: p_left_2_15 + p_right_2_15 + p_down_2_15 + p_up_2_15
   >= 1
 non_overlap_left_2_15: w_var_2 + x_grid_var_2 - x_grid_var_15
   + 774 p_left_2_15 <= 774
 non_overlap_left_rev_2_15: - w_var_2 - x_grid_var_2 + x_grid_var_15
   - 774 p_left_2_15 <= 0
 non_overlap_right_2_15: w_var_15 - x_grid_var_2 + x_grid_var_15
   + 774 p_right_2_15 <= 774
 non_overlap_right_rev_2_15: - w_var_15 + x_grid_var_2 - x_grid_var_15
   - 774 p_right_2_15 <= 0
 non_overlap_down_2_15: h_var_2 + y_grid_var_2 - y_grid_var_15
   + 774 p_down_2_15 <= 774
 non_overlap_down_rev_2_15: - h_var_2 - y_grid_var_2 + y_grid_var_15
   - 774 p_down_2_15 <= 0
 non_overlap_up_2_15: h_var_15 - y_grid_var_2 + y_grid_var_15
   + 774 p_up_2_15 <= 774
 non_overlap_up_rev_2_15: - h_var_15 + y_grid_var_2 - y_grid_var_15
   - 774 p_up_2_15 <= 0
 non_overlap_any_3_4: p_left_3_4 + p_right_3_4 + p_down_3_4 + p_up_3_4
   >= 1
 non_overlap_left_3_4: w_var_3 + x_grid_var_3 - x_grid_var_4
   + 774 p_left_3_4 <= 774
 non_overlap_left_rev_3_4: - w_var_3 - x_grid_var_3 + x_grid_var_4
   - 774 p_left_3_4 <= 0
 non_overlap_right_3_4: w_var_4 - x_grid_var_3 + x_grid_var_4
   + 774 p_right_3_4 <= 774
 non_overlap_right_rev_3_4: - w_var_4 + x_grid_var_3 - x_grid_var_4
   - 774 p_right_3_4 <= 0
 non_overlap_down_3_4: h_var_3 + y_grid_var_3 - y_grid_var_4
   + 774 p_down_3_4 <= 774
 non_overlap_down_rev_3_4: - h_var_3 - y_grid_var_3 + y_grid_var_4
   - 774 p_down_3_4 <= 0
 non_overlap_up_3_4: h_var_4 - y_grid_var_3 + y_grid_var_4 + 774 p_up_3_4
   <= 774
 non_overlap_up_rev_3_4: - h_var_4 + y_grid_var_3 - y_grid_var_4
   - 774 p_up_3_4 <= 0
 non_overlap_any_3_5: p_left_3_5 + p_right_3_5 + p_down_3_5 + p_up_3_5
   >= 1
 non_overlap_left_3_5: w_var_3 + x_grid_var_3 - x_grid_var_5
   + 774 p_left_3_5 <= 774
 non_overlap_left_rev_3_5: - w_var_3 - x_grid_var_3 + x_grid_var_5
   - 774 p_left_3_5 <= 0
 non_overlap_right_3_5: w_var_5 - x_grid_var_3 + x_grid_var_5
   + 774 p_right_3_5 <= 774
 non_overlap_right_rev_3_5: - w_var_5 + x_grid_var_3 - x_grid_var_5
   - 774 p_right_3_5 <= 0
 non_overlap_down_3_5: h_var_3 + y_grid_var_3 - y_grid_var_5
   + 774 p_down_3_5 <= 774
 non_overlap_down_rev_3_5: - h_var_3 - y_grid_var_3 + y_grid_var_5
   - 774 p_down_3_5 <= 0
 non_overlap_up_3_5: h_var_5 - y_grid_var_3 + y_grid_var_5 + 774 p_up_3_5
   <= 774
 non_overlap_up_rev_3_5: - h_var_5 + y_grid_var_3 - y_grid_var_5
   - 774 p_up_3_5 <= 0
 non_overlap_any_3_6: p_left_3_6 + p_right_3_6 + p_down_3_6 + p_up_3_6
   >= 1
 non_overlap_left_3_6: w_var_3 + x_grid_var_3 - x_grid_var_6
   + 774 p_left_3_6 <= 774
 non_overlap_left_rev_3_6: - w_var_3 - x_grid_var_3 + x_grid_var_6
   - 774 p_left_3_6 <= 0
 non_overlap_right_3_6: w_var_6 - x_grid_var_3 + x_grid_var_6
   + 774 p_right_3_6 <= 774
 non_overlap_right_rev_3_6: - w_var_6 + x_grid_var_3 - x_grid_var_6
   - 774 p_right_3_6 <= 0
 non_overlap_down_3_6: h_var_3 + y_grid_var_3 - y_grid_var_6
   + 774 p_down_3_6 <= 774
 non_overlap_down_rev_3_6: - h_var_3 - y_grid_var_3 + y_grid_var_6
   - 774 p_down_3_6 <= 0
 non_overlap_up_3_6: h_var_6 - y_grid_var_3 + y_grid_var_6 + 774 p_up_3_6
   <= 774
 non_overlap_up_rev_3_6: - h_var_6 + y_grid_var_3 - y_grid_var_6
   - 774 p_up_3_6 <= 0
 non_overlap_any_3_7: p_left_3_7 + p_right_3_7 + p_down_3_7 + p_up_3_7
   >= 1
 non_overlap_left_3_7: w_var_3 + x_grid_var_3 - x_grid_var_7
   + 774 p_left_3_7 <= 774
 non_overlap_left_rev_3_7: - w_var_3 - x_grid_var_3 + x_grid_var_7
   - 774 p_left_3_7 <= 0
 non_overlap_right_3_7: w_var_7 - x_grid_var_3 + x_grid_var_7
   + 774 p_right_3_7 <= 774
 non_overlap_right_rev_3_7: - w_var_7 + x_grid_var_3 - x_grid_var_7
   - 774 p_right_3_7 <= 0
 non_overlap_down_3_7: h_var_3 + y_grid_var_3 - y_grid_var_7
   + 774 p_down_3_7 <= 774
 non_overlap_down_rev_3_7: - h_var_3 - y_grid_var_3 + y_grid_var_7
   - 774 p_down_3_7 <= 0
 non_overlap_up_3_7: h_var_7 - y_grid_var_3 + y_grid_var_7 + 774 p_up_3_7
   <= 774
 non_overlap_up_rev_3_7: - h_var_7 + y_grid_var_3 - y_grid_var_7
   - 774 p_up_3_7 <= 0
 non_overlap_any_3_8: p_left_3_8 + p_right_3_8 + p_down_3_8 + p_up_3_8
   >= 1
 non_overlap_left_3_8: w_var_3 + x_grid_var_3 - x_grid_var_8
   + 774 p_left_3_8 <= 774
 non_overlap_left_rev_3_8: - w_var_3 - x_grid_var_3 + x_grid_var_8
   - 774 p_left_3_8 <= 0
 non_overlap_right_3_8: w_var_8 - x_grid_var_3 + x_grid_var_8
   + 774 p_right_3_8 <= 774
 non_overlap_right_rev_3_8: - w_var_8 + x_grid_var_3 - x_grid_var_8
   - 774 p_right_3_8 <= 0
 non_overlap_down_3_8: h_var_3 + y_grid_var_3 - y_grid_var_8
   + 774 p_down_3_8 <= 774
 non_overlap_down_rev_3_8: - h_var_3 - y_grid_var_3 + y_grid_var_8
   - 774 p_down_3_8 <= 0
 non_overlap_up_3_8: h_var_8 - y_grid_var_3 + y_grid_var_8 + 774 p_up_3_8
   <= 774
 non_overlap_up_rev_3_8: - h_var_8 + y_grid_var_3 - y_grid_var_8
   - 774 p_up_3_8 <= 0
 non_overlap_any_3_9: p_left_3_9 + p_right_3_9 + p_down_3_9 + p_up_3_9
   >= 1
 non_overlap_left_3_9: w_var_3 + x_grid_var_3 - x_grid_var_9
   + 774 p_left_3_9 <= 774
 non_overlap_left_rev_3_9: - w_var_3 - x_grid_var_3 + x_grid_var_9
   - 774 p_left_3_9 <= 0
 non_overlap_right_3_9: w_var_9 - x_grid_var_3 + x_grid_var_9
   + 774 p_right_3_9 <= 774
 non_overlap_right_rev_3_9: - w_var_9 + x_grid_var_3 - x_grid_var_9
   - 774 p_right_3_9 <= 0
 non_overlap_down_3_9: h_var_3 + y_grid_var_3 - y_grid_var_9
   + 774 p_down_3_9 <= 774
 non_overlap_down_rev_3_9: - h_var_3 - y_grid_var_3 + y_grid_var_9
   - 774 p_down_3_9 <= 0
 non_overlap_up_3_9: h_var_9 - y_grid_var_3 + y_grid_var_9 + 774 p_up_3_9
   <= 774
 non_overlap_up_rev_3_9: - h_var_9 + y_grid_var_3 - y_grid_var_9
   - 774 p_up_3_9 <= 0
 non_overlap_any_3_10: p_left_3_10 + p_right_3_10 + p_down_3_10 + p_up_3_10
   >= 1
 non_overlap_left_3_10: w_var_3 + x_grid_var_3 - x_grid_var_10
   + 774 p_left_3_10 <= 774
 non_overlap_left_rev_3_10: - w_var_3 - x_grid_var_3 + x_grid_var_10
   - 774 p_left_3_10 <= 0
 non_overlap_right_3_10: w_var_10 - x_grid_var_3 + x_grid_var_10
   + 774 p_right_3_10 <= 774
 non_overlap_right_rev_3_10: - w_var_10 + x_grid_var_3 - x_grid_var_10
   - 774 p_right_3_10 <= 0
 non_overlap_down_3_10: h_var_3 + y_grid_var_3 - y_grid_var_10
   + 774 p_down_3_10 <= 774
 non_overlap_down_rev_3_10: - h_var_3 - y_grid_var_3 + y_grid_var_10
   - 774 p_down_3_10 <= 0
 non_overlap_up_3_10: h_var_10 - y_grid_var_3 + y_grid_var_10
   + 774 p_up_3_10 <= 774
 non_overlap_up_rev_3_10: - h_var_10 + y_grid_var_3 - y_grid_var_10
   - 774 p_up_3_10 <= 0
 non_overlap_any_3_11: p_left_3_11 + p_right_3_11 + p_down_3_11 + p_up_3_11
   >= 1
 non_overlap_left_3_11: w_var_3 + x_grid_var_3 - x_grid_var_11
   + 774 p_left_3_11 <= 774
 non_overlap_left_rev_3_11: - w_var_3 - x_grid_var_3 + x_grid_var_11
   - 774 p_left_3_11 <= 0
 non_overlap_right_3_11: w_var_11 - x_grid_var_3 + x_grid_var_11
   + 774 p_right_3_11 <= 774
 non_overlap_right_rev_3_11: - w_var_11 + x_grid_var_3 - x_grid_var_11
   - 774 p_right_3_11 <= 0
 non_overlap_down_3_11: h_var_3 + y_grid_var_3 - y_grid_var_11
   + 774 p_down_3_11 <= 774
 non_overlap_down_rev_3_11: - h_var_3 - y_grid_var_3 + y_grid_var_11
   - 774 p_down_3_11 <= 0
 non_overlap_up_3_11: h_var_11 - y_grid_var_3 + y_grid_var_11
   + 774 p_up_3_11 <= 774
 non_overlap_up_rev_3_11: - h_var_11 + y_grid_var_3 - y_grid_var_11
   - 774 p_up_3_11 <= 0
 non_overlap_any_3_12: p_left_3_12 + p_right_3_12 + p_down_3_12 + p_up_3_12
   >= 1
 non_overlap_left_3_12: w_var_3 + x_grid_var_3 - x_grid_var_12
   + 774 p_left_3_12 <= 774
 non_overlap_left_rev_3_12: - w_var_3 - x_grid_var_3 + x_grid_var_12
   - 774 p_left_3_12 <= 0
 non_overlap_right_3_12: w_var_12 - x_grid_var_3 + x_grid_var_12
   + 774 p_right_3_12 <= 774
 non_overlap_right_rev_3_12: - w_var_12 + x_grid_var_3 - x_grid_var_12
   - 774 p_right_3_12 <= 0
 non_overlap_down_3_12: h_var_3 + y_grid_var_3 - y_grid_var_12
   + 774 p_down_3_12 <= 774
 non_overlap_down_rev_3_12: - h_var_3 - y_grid_var_3 + y_grid_var_12
   - 774 p_down_3_12 <= 0
 non_overlap_up_3_12: h_var_12 - y_grid_var_3 + y_grid_var_12
   + 774 p_up_3_12 <= 774
 non_overlap_up_rev_3_12: - h_var_12 + y_grid_var_3 - y_grid_var_12
   - 774 p_up_3_12 <= 0
 non_overlap_any_3_13: p_left_3_13 + p_right_3_13 + p_down_3_13 + p_up_3_13
   >= 1
 non_overlap_left_3_13: w_var_3 + x_grid_var_3 - x_grid_var_13
   + 774 p_left_3_13 <= 774
 non_overlap_left_rev_3_13: - w_var_3 - x_grid_var_3 + x_grid_var_13
   - 774 p_left_3_13 <= 0
 non_overlap_right_3_13: w_var_13 - x_grid_var_3 + x_grid_var_13
   + 774 p_right_3_13 <= 774
 non_overlap_right_rev_3_13: - w_var_13 + x_grid_var_3 - x_grid_var_13
   - 774 p_right_3_13 <= 0
 non_overlap_down_3_13: h_var_3 + y_grid_var_3 - y_grid_var_13
   + 774 p_down_3_13 <= 774
 non_overlap_down_rev_3_13: - h_var_3 - y_grid_var_3 + y_grid_var_13
   - 774 p_down_3_13 <= 0
 non_overlap_up_3_13: h_var_13 - y_grid_var_3 + y_grid_var_13
   + 774 p_up_3_13 <= 774
 non_overlap_up_rev_3_13: - h_var_13 + y_grid_var_3 - y_grid_var_13
   - 774 p_up_3_13 <= 0
 non_overlap_any_3_14: p_left_3_14 + p_right_3_14 + p_down_3_14 + p_up_3_14
   >= 1
 non_overlap_left_3_14: w_var_3 + x_grid_var_3 - x_grid_var_14
   + 774 p_left_3_14 <= 774
 non_overlap_left_rev_3_14: - w_var_3 - x_grid_var_3 + x_grid_var_14
   - 774 p_left_3_14 <= 0
 non_overlap_right_3_14: w_var_14 - x_grid_var_3 + x_grid_var_14
   + 774 p_right_3_14 <= 774
 non_overlap_right_rev_3_14: - w_var_14 + x_grid_var_3 - x_grid_var_14
   - 774 p_right_3_14 <= 0
 non_overlap_down_3_14: h_var_3 + y_grid_var_3 - y_grid_var_14
   + 774 p_down_3_14 <= 774
 non_overlap_down_rev_3_14: - h_var_3 - y_grid_var_3 + y_grid_var_14
   - 774 p_down_3_14 <= 0
 non_overlap_up_3_14: h_var_14 - y_grid_var_3 + y_grid_var_14
   + 774 p_up_3_14 <= 774
 non_overlap_up_rev_3_14: - h_var_14 + y_grid_var_3 - y_grid_var_14
   - 774 p_up_3_14 <= 0
 non_overlap_any_3_15: p_left_3_15 + p_right_3_15 + p_down_3_15 + p_up_3_15
   >= 1
 non_overlap_left_3_15: w_var_3 + x_grid_var_3 - x_grid_var_15
   + 774 p_left_3_15 <= 774
 non_overlap_left_rev_3_15: - w_var_3 - x_grid_var_3 + x_grid_var_15
   - 774 p_left_3_15 <= 0
 non_overlap_right_3_15: w_var_15 - x_grid_var_3 + x_grid_var_15
   + 774 p_right_3_15 <= 774
 non_overlap_right_rev_3_15: - w_var_15 + x_grid_var_3 - x_grid_var_15
   - 774 p_right_3_15 <= 0
 non_overlap_down_3_15: h_var_3 + y_grid_var_3 - y_grid_var_15
   + 774 p_down_3_15 <= 774
 non_overlap_down_rev_3_15: - h_var_3 - y_grid_var_3 + y_grid_var_15
   - 774 p_down_3_15 <= 0
 non_overlap_up_3_15: h_var_15 - y_grid_var_3 + y_grid_var_15
   + 774 p_up_3_15 <= 774
 non_overlap_up_rev_3_15: - h_var_15 + y_grid_var_3 - y_grid_var_15
   - 774 p_up_3_15 <= 0
 non_overlap_any_4_5: p_left_4_5 + p_right_4_5 + p_down_4_5 + p_up_4_5
   >= 1
 non_overlap_left_4_5: w_var_4 + x_grid_var_4 - x_grid_var_5
   + 774 p_left_4_5 <= 774
 non_overlap_left_rev_4_5: - w_var_4 - x_grid_var_4 + x_grid_var_5
   - 774 p_left_4_5 <= 0
 non_overlap_right_4_5: w_var_5 - x_grid_var_4 + x_grid_var_5
   + 774 p_right_4_5 <= 774
 non_overlap_right_rev_4_5: - w_var_5 + x_grid_var_4 - x_grid_var_5
   - 774 p_right_4_5 <= 0
 non_overlap_down_4_5: h_var_4 + y_grid_var_4 - y_grid_var_5
   + 774 p_down_4_5 <= 774
 non_overlap_down_rev_4_5: - h_var_4 - y_grid_var_4 + y_grid_var_5
   - 774 p_down_4_5 <= 0
 non_overlap_up_4_5: h_var_5 - y_grid_var_4 + y_grid_var_5 + 774 p_up_4_5
   <= 774
 non_overlap_up_rev_4_5: - h_var_5 + y_grid_var_4 - y_grid_var_5
   - 774 p_up_4_5 <= 0
 non_overlap_any_4_6: p_left_4_6 + p_right_4_6 + p_down_4_6 + p_up_4_6
   >= 1
 non_overlap_left_4_6: w_var_4 + x_grid_var_4 - x_grid_var_6
   + 774 p_left_4_6 <= 774
 non_overlap_left_rev_4_6: - w_var_4 - x_grid_var_4 + x_grid_var_6
   - 774 p_left_4_6 <= 0
 non_overlap_right_4_6: w_var_6 - x_grid_var_4 + x_grid_var_6
   + 774 p_right_4_6 <= 774
 non_overlap_right_rev_4_6: - w_var_6 + x_grid_var_4 - x_grid_var_6
   - 774 p_right_4_6 <= 0
 non_overlap_down_4_6: h_var_4 + y_grid_var_4 - y_grid_var_6
   + 774 p_down_4_6 <= 774
 non_overlap_down_rev_4_6: - h_var_4 - y_grid_var_4 + y_grid_var_6
   - 774 p_down_4_6 <= 0
 non_overlap_up_4_6: h_var_6 - y_grid_var_4 + y_grid_var_6 + 774 p_up_4_6
   <= 774
 non_overlap_up_rev_4_6: - h_var_6 + y_grid_var_4 - y_grid_var_6
   - 774 p_up_4_6 <= 0
 non_overlap_any_4_7: p_left_4_7 + p_right_4_7 + p_down_4_7 + p_up_4_7
   >= 1
 non_overlap_left_4_7: w_var_4 + x_grid_var_4 - x_grid_var_7
   + 774 p_left_4_7 <= 774
 non_overlap_left_rev_4_7: - w_var_4 - x_grid_var_4 + x_grid_var_7
   - 774 p_left_4_7 <= 0
 non_overlap_right_4_7: w_var_7 - x_grid_var_4 + x_grid_var_7
   + 774 p_right_4_7 <= 774
 non_overlap_right_rev_4_7: - w_var_7 + x_grid_var_4 - x_grid_var_7
   - 774 p_right_4_7 <= 0
 non_overlap_down_4_7: h_var_4 + y_grid_var_4 - y_grid_var_7
   + 774 p_down_4_7 <= 774
 non_overlap_down_rev_4_7: - h_var_4 - y_grid_var_4 + y_grid_var_7
   - 774 p_down_4_7 <= 0
 non_overlap_up_4_7: h_var_7 - y_grid_var_4 + y_grid_var_7 + 774 p_up_4_7
   <= 774
 non_overlap_up_rev_4_7: - h_var_7 + y_grid_var_4 - y_grid_var_7
   - 774 p_up_4_7 <= 0
 non_overlap_any_4_8: p_left_4_8 + p_right_4_8 + p_down_4_8 + p_up_4_8
   >= 1
 non_overlap_left_4_8: w_var_4 + x_grid_var_4 - x_grid_var_8
   + 774 p_left_4_8 <= 774
 non_overlap_left_rev_4_8: - w_var_4 - x_grid_var_4 + x_grid_var_8
   - 774 p_left_4_8 <= 0
 non_overlap_right_4_8: w_var_8 - x_grid_var_4 + x_grid_var_8
   + 774 p_right_4_8 <= 774
 non_overlap_right_rev_4_8: - w_var_8 + x_grid_var_4 - x_grid_var_8
   - 774 p_right_4_8 <= 0
 non_overlap_down_4_8: h_var_4 + y_grid_var_4 - y_grid_var_8
   + 774 p_down_4_8 <= 774
 non_overlap_down_rev_4_8: - h_var_4 - y_grid_var_4 + y_grid_var_8
   - 774 p_down_4_8 <= 0
 non_overlap_up_4_8: h_var_8 - y_grid_var_4 + y_grid_var_8 + 774 p_up_4_8
   <= 774
 non_overlap_up_rev_4_8: - h_var_8 + y_grid_var_4 - y_grid_var_8
   - 774 p_up_4_8 <= 0
 non_overlap_any_4_9: p_left_4_9 + p_right_4_9 + p_down_4_9 + p_up_4_9
   >= 1
 non_overlap_left_4_9: w_var_4 + x_grid_var_4 - x_grid_var_9
   + 774 p_left_4_9 <= 774
 non_overlap_left_rev_4_9: - w_var_4 - x_grid_var_4 + x_grid_var_9
   - 774 p_left_4_9 <= 0
 non_overlap_right_4_9: w_var_9 - x_grid_var_4 + x_grid_var_9
   + 774 p_right_4_9 <= 774
 non_overlap_right_rev_4_9: - w_var_9 + x_grid_var_4 - x_grid_var_9
   - 774 p_right_4_9 <= 0
 non_overlap_down_4_9: h_var_4 + y_grid_var_4 - y_grid_var_9
   + 774 p_down_4_9 <= 774
 non_overlap_down_rev_4_9: - h_var_4 - y_grid_var_4 + y_grid_var_9
   - 774 p_down_4_9 <= 0
 non_overlap_up_4_9: h_var_9 - y_grid_var_4 + y_grid_var_9 + 774 p_up_4_9
   <= 774
 non_overlap_up_rev_4_9: - h_var_9 + y_grid_var_4 - y_grid_var_9
   - 774 p_up_4_9 <= 0
 non_overlap_any_4_10: p_left_4_10 + p_right_4_10 + p_down_4_10 + p_up_4_10
   >= 1
 non_overlap_left_4_10: w_var_4 + x_grid_var_4 - x_grid_var_10
   + 774 p_left_4_10 <= 774
 non_overlap_left_rev_4_10: - w_var_4 - x_grid_var_4 + x_grid_var_10
   - 774 p_left_4_10 <= 0
 non_overlap_right_4_10: w_var_10 - x_grid_var_4 + x_grid_var_10
   + 774 p_right_4_10 <= 774
 non_overlap_right_rev_4_10: - w_var_10 + x_grid_var_4 - x_grid_var_10
   - 774 p_right_4_10 <= 0
 non_overlap_down_4_10: h_var_4 + y_grid_var_4 - y_grid_var_10
   + 774 p_down_4_10 <= 774
 non_overlap_down_rev_4_10: - h_var_4 - y_grid_var_4 + y_grid_var_10
   - 774 p_down_4_10 <= 0
 non_overlap_up_4_10: h_var_10 - y_grid_var_4 + y_grid_var_10
   + 774 p_up_4_10 <= 774
 non_overlap_up_rev_4_10: - h_var_10 + y_grid_var_4 - y_grid_var_10
   - 774 p_up_4_10 <= 0
 non_overlap_any_4_11: p_left_4_11 + p_right_4_11 + p_down_4_11 + p_up_4_11
   >= 1
 non_overlap_left_4_11: w_var_4 + x_grid_var_4 - x_grid_var_11
   + 774 p_left_4_11 <= 774
 non_overlap_left_rev_4_11: - w_var_4 - x_grid_var_4 + x_grid_var_11
   - 774 p_left_4_11 <= 0
 non_overlap_right_4_11: w_var_11 - x_grid_var_4 + x_grid_var_11
   + 774 p_right_4_11 <= 774
 non_overlap_right_rev_4_11: - w_var_11 + x_grid_var_4 - x_grid_var_11
   - 774 p_right_4_11 <= 0
 non_overlap_down_4_11: h_var_4 + y_grid_var_4 - y_grid_var_11
   + 774 p_down_4_11 <= 774
 non_overlap_down_rev_4_11: - h_var_4 - y_grid_var_4 + y_grid_var_11
   - 774 p_down_4_11 <= 0
 non_overlap_up_4_11: h_var_11 - y_grid_var_4 + y_grid_var_11
   + 774 p_up_4_11 <= 774
 non_overlap_up_rev_4_11: - h_var_11 + y_grid_var_4 - y_grid_var_11
   - 774 p_up_4_11 <= 0
 non_overlap_any_4_12: p_left_4_12 + p_right_4_12 + p_down_4_12 + p_up_4_12
   >= 1
 non_overlap_left_4_12: w_var_4 + x_grid_var_4 - x_grid_var_12
   + 774 p_left_4_12 <= 774
 non_overlap_left_rev_4_12: - w_var_4 - x_grid_var_4 + x_grid_var_12
   - 774 p_left_4_12 <= 0
 non_overlap_right_4_12: w_var_12 - x_grid_var_4 + x_grid_var_12
   + 774 p_right_4_12 <= 774
 non_overlap_right_rev_4_12: - w_var_12 + x_grid_var_4 - x_grid_var_12
   - 774 p_right_4_12 <= 0
 non_overlap_down_4_12: h_var_4 + y_grid_var_4 - y_grid_var_12
   + 774 p_down_4_12 <= 774
 non_overlap_down_rev_4_12: - h_var_4 - y_grid_var_4 + y_grid_var_12
   - 774 p_down_4_12 <= 0
 non_overlap_up_4_12: h_var_12 - y_grid_var_4 + y_grid_var_12
   + 774 p_up_4_12 <= 774
 non_overlap_up_rev_4_12: - h_var_12 + y_grid_var_4 - y_grid_var_12
   - 774 p_up_4_12 <= 0
 non_overlap_any_4_13: p_left_4_13 + p_right_4_13 + p_down_4_13 + p_up_4_13
   >= 1
 non_overlap_left_4_13: w_var_4 + x_grid_var_4 - x_grid_var_13
   + 774 p_left_4_13 <= 774
 non_overlap_left_rev_4_13: - w_var_4 - x_grid_var_4 + x_grid_var_13
   - 774 p_left_4_13 <= 0
 non_overlap_right_4_13: w_var_13 - x_grid_var_4 + x_grid_var_13
   + 774 p_right_4_13 <= 774
 non_overlap_right_rev_4_13: - w_var_13 + x_grid_var_4 - x_grid_var_13
   - 774 p_right_4_13 <= 0
 non_overlap_down_4_13: h_var_4 + y_grid_var_4 - y_grid_var_13
   + 774 p_down_4_13 <= 774
 non_overlap_down_rev_4_13: - h_var_4 - y_grid_var_4 + y_grid_var_13
   - 774 p_down_4_13 <= 0
 non_overlap_up_4_13: h_var_13 - y_grid_var_4 + y_grid_var_13
   + 774 p_up_4_13 <= 774
 non_overlap_up_rev_4_13: - h_var_13 + y_grid_var_4 - y_grid_var_13
   - 774 p_up_4_13 <= 0
 non_overlap_any_4_14: p_left_4_14 + p_right_4_14 + p_down_4_14 + p_up_4_14
   >= 1
 non_overlap_left_4_14: w_var_4 + x_grid_var_4 - x_grid_var_14
   + 774 p_left_4_14 <= 774
 non_overlap_left_rev_4_14: - w_var_4 - x_grid_var_4 + x_grid_var_14
   - 774 p_left_4_14 <= 0
 non_overlap_right_4_14: w_var_14 - x_grid_var_4 + x_grid_var_14
   + 774 p_right_4_14 <= 774
 non_overlap_right_rev_4_14: - w_var_14 + x_grid_var_4 - x_grid_var_14
   - 774 p_right_4_14 <= 0
 non_overlap_down_4_14: h_var_4 + y_grid_var_4 - y_grid_var_14
   + 774 p_down_4_14 <= 774
 non_overlap_down_rev_4_14: - h_var_4 - y_grid_var_4 + y_grid_var_14
   - 774 p_down_4_14 <= 0
 non_overlap_up_4_14: h_var_14 - y_grid_var_4 + y_grid_var_14
   + 774 p_up_4_14 <= 774
 non_overlap_up_rev_4_14: - h_var_14 + y_grid_var_4 - y_grid_var_14
   - 774 p_up_4_14 <= 0
 non_overlap_any_4_15: p_left_4_15 + p_right_4_15 + p_down_4_15 + p_up_4_15
   >= 1
 non_overlap_left_4_15: w_var_4 + x_grid_var_4 - x_grid_var_15
   + 774 p_left_4_15 <= 774
 non_overlap_left_rev_4_15: - w_var_4 - x_grid_var_4 + x_grid_var_15
   - 774 p_left_4_15 <= 0
 non_overlap_right_4_15: w_var_15 - x_grid_var_4 + x_grid_var_15
   + 774 p_right_4_15 <= 774
 non_overlap_right_rev_4_15: - w_var_15 + x_grid_var_4 - x_grid_var_15
   - 774 p_right_4_15 <= 0
 non_overlap_down_4_15: h_var_4 + y_grid_var_4 - y_grid_var_15
   + 774 p_down_4_15 <= 774
 non_overlap_down_rev_4_15: - h_var_4 - y_grid_var_4 + y_grid_var_15
   - 774 p_down_4_15 <= 0
 non_overlap_up_4_15: h_var_15 - y_grid_var_4 + y_grid_var_15
   + 774 p_up_4_15 <= 774
 non_overlap_up_rev_4_15: - h_var_15 + y_grid_var_4 - y_grid_var_15
   - 774 p_up_4_15 <= 0
 non_overlap_any_5_6: p_left_5_6 + p_right_5_6 + p_down_5_6 + p_up_5_6
   >= 1
 non_overlap_left_5_6: w_var_5 + x_grid_var_5 - x_grid_var_6
   + 774 p_left_5_6 <= 774
 non_overlap_left_rev_5_6: - w_var_5 - x_grid_var_5 + x_grid_var_6
   - 774 p_left_5_6 <= 0
 non_overlap_right_5_6: w_var_6 - x_grid_var_5 + x_grid_var_6
   + 774 p_right_5_6 <= 774
 non_overlap_right_rev_5_6: - w_var_6 + x_grid_var_5 - x_grid_var_6
   - 774 p_right_5_6 <= 0
 non_overlap_down_5_6: h_var_5 + y_grid_var_5 - y_grid_var_6
   + 774 p_down_5_6 <= 774
 non_overlap_down_rev_5_6: - h_var_5 - y_grid_var_5 + y_grid_var_6
   - 774 p_down_5_6 <= 0
 non_overlap_up_5_6: h_var_6 - y_grid_var_5 + y_grid_var_6 + 774 p_up_5_6
   <= 774
 non_overlap_up_rev_5_6: - h_var_6 + y_grid_var_5 - y_grid_var_6
   - 774 p_up_5_6 <= 0
 non_overlap_any_5_7: p_left_5_7 + p_right_5_7 + p_down_5_7 + p_up_5_7
   >= 1
 non_overlap_left_5_7: w_var_5 + x_grid_var_5 - x_grid_var_7
   + 774 p_left_5_7 <= 774
 non_overlap_left_rev_5_7: - w_var_5 - x_grid_var_5 + x_grid_var_7
   - 774 p_left_5_7 <= 0
 non_overlap_right_5_7: w_var_7 - x_grid_var_5 + x_grid_var_7
   + 774 p_right_5_7 <= 774
 non_overlap_right_rev_5_7: - w_var_7 + x_grid_var_5 - x_grid_var_7
   - 774 p_right_5_7 <= 0
 non_overlap_down_5_7: h_var_5 + y_grid_var_5 - y_grid_var_7
   + 774 p_down_5_7 <= 774
 non_overlap_down_rev_5_7: - h_var_5 - y_grid_var_5 + y_grid_var_7
   - 774 p_down_5_7 <= 0
 non_overlap_up_5_7: h_var_7 - y_grid_var_5 + y_grid_var_7 + 774 p_up_5_7
   <= 774
 non_overlap_up_rev_5_7: - h_var_7 + y_grid_var_5 - y_grid_var_7
   - 774 p_up_5_7 <= 0
 non_overlap_any_5_8: p_left_5_8 + p_right_5_8 + p_down_5_8 + p_up_5_8
   >= 1
 non_overlap_left_5_8: w_var_5 + x_grid_var_5 - x_grid_var_8
   + 774 p_left_5_8 <= 774
 non_overlap_left_rev_5_8: - w_var_5 - x_grid_var_5 + x_grid_var_8
   - 774 p_left_5_8 <= 0
 non_overlap_right_5_8: w_var_8 - x_grid_var_5 + x_grid_var_8
   + 774 p_right_5_8 <= 774
 non_overlap_right_rev_5_8: - w_var_8 + x_grid_var_5 - x_grid_var_8
   - 774 p_right_5_8 <= 0
 non_overlap_down_5_8: h_var_5 + y_grid_var_5 - y_grid_var_8
   + 774 p_down_5_8 <= 774
 non_overlap_down_rev_5_8: - h_var_5 - y_grid_var_5 + y_grid_var_8
   - 774 p_down_5_8 <= 0
 non_overlap_up_5_8: h_var_8 - y_grid_var_5 + y_grid_var_8 + 774 p_up_5_8
   <= 774
 non_overlap_up_rev_5_8: - h_var_8 + y_grid_var_5 - y_grid_var_8
   - 774 p_up_5_8 <= 0
 non_overlap_any_5_9: p_left_5_9 + p_right_5_9 + p_down_5_9 + p_up_5_9
   >= 1
 non_overlap_left_5_9: w_var_5 + x_grid_var_5 - x_grid_var_9
   + 774 p_left_5_9 <= 774
 non_overlap_left_rev_5_9: - w_var_5 - x_grid_var_5 + x_grid_var_9
   - 774 p_left_5_9 <= 0
 non_overlap_right_5_9: w_var_9 - x_grid_var_5 + x_grid_var_9
   + 774 p_right_5_9 <= 774
 non_overlap_right_rev_5_9: - w_var_9 + x_grid_var_5 - x_grid_var_9
   - 774 p_right_5_9 <= 0
 non_overlap_down_5_9: h_var_5 + y_grid_var_5 - y_grid_var_9
   + 774 p_down_5_9 <= 774
 non_overlap_down_rev_5_9: - h_var_5 - y_grid_var_5 + y_grid_var_9
   - 774 p_down_5_9 <= 0
 non_overlap_up_5_9: h_var_9 - y_grid_var_5 + y_grid_var_9 + 774 p_up_5_9
   <= 774
 non_overlap_up_rev_5_9: - h_var_9 + y_grid_var_5 - y_grid_var_9
   - 774 p_up_5_9 <= 0
 non_overlap_any_5_10: p_left_5_10 + p_right_5_10 + p_down_5_10 + p_up_5_10
   >= 1
 non_overlap_left_5_10: w_var_5 + x_grid_var_5 - x_grid_var_10
   + 774 p_left_5_10 <= 774
 non_overlap_left_rev_5_10: - w_var_5 - x_grid_var_5 + x_grid_var_10
   - 774 p_left_5_10 <= 0
 non_overlap_right_5_10: w_var_10 - x_grid_var_5 + x_grid_var_10
   + 774 p_right_5_10 <= 774
 non_overlap_right_rev_5_10: - w_var_10 + x_grid_var_5 - x_grid_var_10
   - 774 p_right_5_10 <= 0
 non_overlap_down_5_10: h_var_5 + y_grid_var_5 - y_grid_var_10
   + 774 p_down_5_10 <= 774
 non_overlap_down_rev_5_10: - h_var_5 - y_grid_var_5 + y_grid_var_10
   - 774 p_down_5_10 <= 0
 non_overlap_up_5_10: h_var_10 - y_grid_var_5 + y_grid_var_10
   + 774 p_up_5_10 <= 774
 non_overlap_up_rev_5_10: - h_var_10 + y_grid_var_5 - y_grid_var_10
   - 774 p_up_5_10 <= 0
 non_overlap_any_5_11: p_left_5_11 + p_right_5_11 + p_down_5_11 + p_up_5_11
   >= 1
 non_overlap_left_5_11: w_var_5 + x_grid_var_5 - x_grid_var_11
   + 774 p_left_5_11 <= 774
 non_overlap_left_rev_5_11: - w_var_5 - x_grid_var_5 + x_grid_var_11
   - 774 p_left_5_11 <= 0
 non_overlap_right_5_11: w_var_11 - x_grid_var_5 + x_grid_var_11
   + 774 p_right_5_11 <= 774
 non_overlap_right_rev_5_11: - w_var_11 + x_grid_var_5 - x_grid_var_11
   - 774 p_right_5_11 <= 0
 non_overlap_down_5_11: h_var_5 + y_grid_var_5 - y_grid_var_11
   + 774 p_down_5_11 <= 774
 non_overlap_down_rev_5_11: - h_var_5 - y_grid_var_5 + y_grid_var_11
   - 774 p_down_5_11 <= 0
 non_overlap_up_5_11: h_var_11 - y_grid_var_5 + y_grid_var_11
   + 774 p_up_5_11 <= 774
 non_overlap_up_rev_5_11: - h_var_11 + y_grid_var_5 - y_grid_var_11
   - 774 p_up_5_11 <= 0
 non_overlap_any_5_12: p_left_5_12 + p_right_5_12 + p_down_5_12 + p_up_5_12
   >= 1
 non_overlap_left_5_12: w_var_5 + x_grid_var_5 - x_grid_var_12
   + 774 p_left_5_12 <= 774
 non_overlap_left_rev_5_12: - w_var_5 - x_grid_var_5 + x_grid_var_12
   - 774 p_left_5_12 <= 0
 non_overlap_right_5_12: w_var_12 - x_grid_var_5 + x_grid_var_12
   + 774 p_right_5_12 <= 774
 non_overlap_right_rev_5_12: - w_var_12 + x_grid_var_5 - x_grid_var_12
   - 774 p_right_5_12 <= 0
 non_overlap_down_5_12: h_var_5 + y_grid_var_5 - y_grid_var_12
   + 774 p_down_5_12 <= 774
 non_overlap_down_rev_5_12: - h_var_5 - y_grid_var_5 + y_grid_var_12
   - 774 p_down_5_12 <= 0
 non_overlap_up_5_12: h_var_12 - y_grid_var_5 + y_grid_var_12
   + 774 p_up_5_12 <= 774
 non_overlap_up_rev_5_12: - h_var_12 + y_grid_var_5 - y_grid_var_12
   - 774 p_up_5_12 <= 0
 non_overlap_any_5_13: p_left_5_13 + p_right_5_13 + p_down_5_13 + p_up_5_13
   >= 1
 non_overlap_left_5_13: w_var_5 + x_grid_var_5 - x_grid_var_13
   + 774 p_left_5_13 <= 774
 non_overlap_left_rev_5_13: - w_var_5 - x_grid_var_5 + x_grid_var_13
   - 774 p_left_5_13 <= 0
 non_overlap_right_5_13: w_var_13 - x_grid_var_5 + x_grid_var_13
   + 774 p_right_5_13 <= 774
 non_overlap_right_rev_5_13: - w_var_13 + x_grid_var_5 - x_grid_var_13
   - 774 p_right_5_13 <= 0
 non_overlap_down_5_13: h_var_5 + y_grid_var_5 - y_grid_var_13
   + 774 p_down_5_13 <= 774
 non_overlap_down_rev_5_13: - h_var_5 - y_grid_var_5 + y_grid_var_13
   - 774 p_down_5_13 <= 0
 non_overlap_up_5_13: h_var_13 - y_grid_var_5 + y_grid_var_13
   + 774 p_up_5_13 <= 774
 non_overlap_up_rev_5_13: - h_var_13 + y_grid_var_5 - y_grid_var_13
   - 774 p_up_5_13 <= 0
 non_overlap_any_5_14: p_left_5_14 + p_right_5_14 + p_down_5_14 + p_up_5_14
   >= 1
 non_overlap_left_5_14: w_var_5 + x_grid_var_5 - x_grid_var_14
   + 774 p_left_5_14 <= 774
 non_overlap_left_rev_5_14: - w_var_5 - x_grid_var_5 + x_grid_var_14
   - 774 p_left_5_14 <= 0
 non_overlap_right_5_14: w_var_14 - x_grid_var_5 + x_grid_var_14
   + 774 p_right_5_14 <= 774
 non_overlap_right_rev_5_14: - w_var_14 + x_grid_var_5 - x_grid_var_14
   - 774 p_right_5_14 <= 0
 non_overlap_down_5_14: h_var_5 + y_grid_var_5 - y_grid_var_14
   + 774 p_down_5_14 <= 774
 non_overlap_down_rev_5_14: - h_var_5 - y_grid_var_5 + y_grid_var_14
   - 774 p_down_5_14 <= 0
 non_overlap_up_5_14: h_var_14 - y_grid_var_5 + y_grid_var_14
   + 774 p_up_5_14 <= 774
 non_overlap_up_rev_5_14: - h_var_14 + y_grid_var_5 - y_grid_var_14
   - 774 p_up_5_14 <= 0
 non_overlap_any_5_15: p_left_5_15 + p_right_5_15 + p_down_5_15 + p_up_5_15
   >= 1
 non_overlap_left_5_15: w_var_5 + x_grid_var_5 - x_grid_var_15
   + 774 p_left_5_15 <= 774
 non_overlap_left_rev_5_15: - w_var_5 - x_grid_var_5 + x_grid_var_15
   - 774 p_left_5_15 <= 0
 non_overlap_right_5_15: w_var_15 - x_grid_var_5 + x_grid_var_15
   + 774 p_right_5_15 <= 774
 non_overlap_right_rev_5_15: - w_var_15 + x_grid_var_5 - x_grid_var_15
   - 774 p_right_5_15 <= 0
 non_overlap_down_5_15: h_var_5 + y_grid_var_5 - y_grid_var_15
   + 774 p_down_5_15 <= 774
 non_overlap_down_rev_5_15: - h_var_5 - y_grid_var_5 + y_grid_var_15
   - 774 p_down_5_15 <= 0
 non_overlap_up_5_15: h_var_15 - y_grid_var_5 + y_grid_var_15
   + 774 p_up_5_15 <= 774
 non_overlap_up_rev_5_15: - h_var_15 + y_grid_var_5 - y_grid_var_15
   - 774 p_up_5_15 <= 0
 non_overlap_any_6_7: p_left_6_7 + p_right_6_7 + p_down_6_7 + p_up_6_7
   >= 1
 non_overlap_left_6_7: w_var_6 + x_grid_var_6 - x_grid_var_7
   + 774 p_left_6_7 <= 774
 non_overlap_left_rev_6_7: - w_var_6 - x_grid_var_6 + x_grid_var_7
   - 774 p_left_6_7 <= 0
 non_overlap_right_6_7: w_var_7 - x_grid_var_6 + x_grid_var_7
   + 774 p_right_6_7 <= 774
 non_overlap_right_rev_6_7: - w_var_7 + x_grid_var_6 - x_grid_var_7
   - 774 p_right_6_7 <= 0
 non_overlap_down_6_7: h_var_6 + y_grid_var_6 - y_grid_var_7
   + 774 p_down_6_7 <= 774
 non_overlap_down_rev_6_7: - h_var_6 - y_grid_var_6 + y_grid_var_7
   - 774 p_down_6_7 <= 0
 non_overlap_up_6_7: h_var_7 - y_grid_var_6 + y_grid_var_7 + 774 p_up_6_7
   <= 774
 non_overlap_up_rev_6_7: - h_var_7 + y_grid_var_6 - y_grid_var_7
   - 774 p_up_6_7 <= 0
 non_overlap_any_6_8: p_left_6_8 + p_right_6_8 + p_down_6_8 + p_up_6_8
   >= 1
 non_overlap_left_6_8: w_var_6 + x_grid_var_6 - x_grid_var_8
   + 774 p_left_6_8 <= 774
 non_overlap_left_rev_6_8: - w_var_6 - x_grid_var_6 + x_grid_var_8
   - 774 p_left_6_8 <= 0
 non_overlap_right_6_8: w_var_8 - x_grid_var_6 + x_grid_var_8
   + 774 p_right_6_8 <= 774
 non_overlap_right_rev_6_8: - w_var_8 + x_grid_var_6 - x_grid_var_8
   - 774 p_right_6_8 <= 0
 non_overlap_down_6_8: h_var_6 + y_grid_var_6 - y_grid_var_8
   + 774 p_down_6_8 <= 774
 non_overlap_down_rev_6_8: - h_var_6 - y_grid_var_6 + y_grid_var_8
   - 774 p_down_6_8 <= 0
 non_overlap_up_6_8: h_var_8 - y_grid_var_6 + y_grid_var_8 + 774 p_up_6_8
   <= 774
 non_overlap_up_rev_6_8: - h_var_8 + y_grid_var_6 - y_grid_var_8
   - 774 p_up_6_8 <= 0
 non_overlap_any_6_9: p_left_6_9 + p_right_6_9 + p_down_6_9 + p_up_6_9
   >= 1
 non_overlap_left_6_9: w_var_6 + x_grid_var_6 - x_grid_var_9
   + 774 p_left_6_9 <= 774
 non_overlap_left_rev_6_9: - w_var_6 - x_grid_var_6 + x_grid_var_9
   - 774 p_left_6_9 <= 0
 non_overlap_right_6_9: w_var_9 - x_grid_var_6 + x_grid_var_9
   + 774 p_right_6_9 <= 774
 non_overlap_right_rev_6_9: - w_var_9 + x_grid_var_6 - x_grid_var_9
   - 774 p_right_6_9 <= 0
 non_overlap_down_6_9: h_var_6 + y_grid_var_6 - y_grid_var_9
   + 774 p_down_6_9 <= 774
 non_overlap_down_rev_6_9: - h_var_6 - y_grid_var_6 + y_grid_var_9
   - 774 p_down_6_9 <= 0
 non_overlap_up_6_9: h_var_9 - y_grid_var_6 + y_grid_var_9 + 774 p_up_6_9
   <= 774
 non_overlap_up_rev_6_9: - h_var_9 + y_grid_var_6 - y_grid_var_9
   - 774 p_up_6_9 <= 0
 non_overlap_any_6_10: p_left_6_10 + p_right_6_10 + p_down_6_10 + p_up_6_10
   >= 1
 non_overlap_left_6_10: w_var_6 + x_grid_var_6 - x_grid_var_10
   + 774 p_left_6_10 <= 774
 non_overlap_left_rev_6_10: - w_var_6 - x_grid_var_6 + x_grid_var_10
   - 774 p_left_6_10 <= 0
 non_overlap_right_6_10: w_var_10 - x_grid_var_6 + x_grid_var_10
   + 774 p_right_6_10 <= 774
 non_overlap_right_rev_6_10: - w_var_10 + x_grid_var_6 - x_grid_var_10
   - 774 p_right_6_10 <= 0
 non_overlap_down_6_10: h_var_6 + y_grid_var_6 - y_grid_var_10
   + 774 p_down_6_10 <= 774
 non_overlap_down_rev_6_10: - h_var_6 - y_grid_var_6 + y_grid_var_10
   - 774 p_down_6_10 <= 0
 non_overlap_up_6_10: h_var_10 - y_grid_var_6 + y_grid_var_10
   + 774 p_up_6_10 <= 774
 non_overlap_up_rev_6_10: - h_var_10 + y_grid_var_6 - y_grid_var_10
   - 774 p_up_6_10 <= 0
 non_overlap_any_6_11: p_left_6_11 + p_right_6_11 + p_down_6_11 + p_up_6_11
   >= 1
 non_overlap_left_6_11: w_var_6 + x_grid_var_6 - x_grid_var_11
   + 774 p_left_6_11 <= 774
 non_overlap_left_rev_6_11: - w_var_6 - x_grid_var_6 + x_grid_var_11
   - 774 p_left_6_11 <= 0
 non_overlap_right_6_11: w_var_11 - x_grid_var_6 + x_grid_var_11
   + 774 p_right_6_11 <= 774
 non_overlap_right_rev_6_11: - w_var_11 + x_grid_var_6 - x_grid_var_11
   - 774 p_right_6_11 <= 0
 non_overlap_down_6_11: h_var_6 + y_grid_var_6 - y_grid_var_11
   + 774 p_down_6_11 <= 774
 non_overlap_down_rev_6_11: - h_var_6 - y_grid_var_6 + y_grid_var_11
   - 774 p_down_6_11 <= 0
 non_overlap_up_6_11: h_var_11 - y_grid_var_6 + y_grid_var_11
   + 774 p_up_6_11 <= 774
 non_overlap_up_rev_6_11: - h_var_11 + y_grid_var_6 - y_grid_var_11
   - 774 p_up_6_11 <= 0
 non_overlap_any_6_12: p_left_6_12 + p_right_6_12 + p_down_6_12 + p_up_6_12
   >= 1
 non_overlap_left_6_12: w_var_6 + x_grid_var_6 - x_grid_var_12
   + 774 p_left_6_12 <= 774
 non_overlap_left_rev_6_12: - w_var_6 - x_grid_var_6 + x_grid_var_12
   - 774 p_left_6_12 <= 0
 non_overlap_right_6_12: w_var_12 - x_grid_var_6 + x_grid_var_12
   + 774 p_right_6_12 <= 774
 non_overlap_right_rev_6_12: - w_var_12 + x_grid_var_6 - x_grid_var_12
   - 774 p_right_6_12 <= 0
 non_overlap_down_6_12: h_var_6 + y_grid_var_6 - y_grid_var_12
   + 774 p_down_6_12 <= 774
 non_overlap_down_rev_6_12: - h_var_6 - y_grid_var_6 + y_grid_var_12
   - 774 p_down_6_12 <= 0
 non_overlap_up_6_12: h_var_12 - y_grid_var_6 + y_grid_var_12
   + 774 p_up_6_12 <= 774
 non_overlap_up_rev_6_12: - h_var_12 + y_grid_var_6 - y_grid_var_12
   - 774 p_up_6_12 <= 0
 non_overlap_any_6_13: p_left_6_13 + p_right_6_13 + p_down_6_13 + p_up_6_13
   >= 1
 non_overlap_left_6_13: w_var_6 + x_grid_var_6 - x_grid_var_13
   + 774 p_left_6_13 <= 774
 non_overlap_left_rev_6_13: - w_var_6 - x_grid_var_6 + x_grid_var_13
   - 774 p_left_6_13 <= 0
 non_overlap_right_6_13: w_var_13 - x_grid_var_6 + x_grid_var_13
   + 774 p_right_6_13 <= 774
 non_overlap_right_rev_6_13: - w_var_13 + x_grid_var_6 - x_grid_var_13
   - 774 p_right_6_13 <= 0
 non_overlap_down_6_13: h_var_6 + y_grid_var_6 - y_grid_var_13
   + 774 p_down_6_13 <= 774
 non_overlap_down_rev_6_13: - h_var_6 - y_grid_var_6 + y_grid_var_13
   - 774 p_down_6_13 <= 0
 non_overlap_up_6_13: h_var_13 - y_grid_var_6 + y_grid_var_13
   + 774 p_up_6_13 <= 774
 non_overlap_up_rev_6_13: - h_var_13 + y_grid_var_6 - y_grid_var_13
   - 774 p_up_6_13 <= 0
 non_overlap_any_6_14: p_left_6_14 + p_right_6_14 + p_down_6_14 + p_up_6_14
   >= 1
 non_overlap_left_6_14: w_var_6 + x_grid_var_6 - x_grid_var_14
   + 774 p_left_6_14 <= 774
 non_overlap_left_rev_6_14: - w_var_6 - x_grid_var_6 + x_grid_var_14
   - 774 p_left_6_14 <= 0
 non_overlap_right_6_14: w_var_14 - x_grid_var_6 + x_grid_var_14
   + 774 p_right_6_14 <= 774
 non_overlap_right_rev_6_14: - w_var_14 + x_grid_var_6 - x_grid_var_14
   - 774 p_right_6_14 <= 0
 non_overlap_down_6_14: h_var_6 + y_grid_var_6 - y_grid_var_14
   + 774 p_down_6_14 <= 774
 non_overlap_down_rev_6_14: - h_var_6 - y_grid_var_6 + y_grid_var_14
   - 774 p_down_6_14 <= 0
 non_overlap_up_6_14: h_var_14 - y_grid_var_6 + y_grid_var_14
   + 774 p_up_6_14 <= 774
 non_overlap_up_rev_6_14: - h_var_14 + y_grid_var_6 - y_grid_var_14
   - 774 p_up_6_14 <= 0
 non_overlap_any_6_15: p_left_6_15 + p_right_6_15 + p_down_6_15 + p_up_6_15
   >= 1
 non_overlap_left_6_15: w_var_6 + x_grid_var_6 - x_grid_var_15
   + 774 p_left_6_15 <= 774
 non_overlap_left_rev_6_15: - w_var_6 - x_grid_var_6 + x_grid_var_15
   - 774 p_left_6_15 <= 0
 non_overlap_right_6_15: w_var_15 - x_grid_var_6 + x_grid_var_15
   + 774 p_right_6_15 <= 774
 non_overlap_right_rev_6_15: - w_var_15 + x_grid_var_6 - x_grid_var_15
   - 774 p_right_6_15 <= 0
 non_overlap_down_6_15: h_var_6 + y_grid_var_6 - y_grid_var_15
   + 774 p_down_6_15 <= 774
 non_overlap_down_rev_6_15: - h_var_6 - y_grid_var_6 + y_grid_var_15
   - 774 p_down_6_15 <= 0
 non_overlap_up_6_15: h_var_15 - y_grid_var_6 + y_grid_var_15
   + 774 p_up_6_15 <= 774
 non_overlap_up_rev_6_15: - h_var_15 + y_grid_var_6 - y_grid_var_15
   - 774 p_up_6_15 <= 0
 non_overlap_any_7_8: p_left_7_8 + p_right_7_8 + p_down_7_8 + p_up_7_8
   >= 1
 non_overlap_left_7_8: w_var_7 + x_grid_var_7 - x_grid_var_8
   + 774 p_left_7_8 <= 774
 non_overlap_left_rev_7_8: - w_var_7 - x_grid_var_7 + x_grid_var_8
   - 774 p_left_7_8 <= 0
 non_overlap_right_7_8: w_var_8 - x_grid_var_7 + x_grid_var_8
   + 774 p_right_7_8 <= 774
 non_overlap_right_rev_7_8: - w_var_8 + x_grid_var_7 - x_grid_var_8
   - 774 p_right_7_8 <= 0
 non_overlap_down_7_8: h_var_7 + y_grid_var_7 - y_grid_var_8
   + 774 p_down_7_8 <= 774
 non_overlap_down_rev_7_8: - h_var_7 - y_grid_var_7 + y_grid_var_8
   - 774 p_down_7_8 <= 0
 non_overlap_up_7_8: h_var_8 - y_grid_var_7 + y_grid_var_8 + 774 p_up_7_8
   <= 774
 non_overlap_up_rev_7_8: - h_var_8 + y_grid_var_7 - y_grid_var_8
   - 774 p_up_7_8 <= 0
 non_overlap_any_7_9: p_left_7_9 + p_right_7_9 + p_down_7_9 + p_up_7_9
   >= 1
 non_overlap_left_7_9: w_var_7 + x_grid_var_7 - x_grid_var_9
   + 774 p_left_7_9 <= 774
 non_overlap_left_rev_7_9: - w_var_7 - x_grid_var_7 + x_grid_var_9
   - 774 p_left_7_9 <= 0
 non_overlap_right_7_9: w_var_9 - x_grid_var_7 + x_grid_var_9
   + 774 p_right_7_9 <= 774
 non_overlap_right_rev_7_9: - w_var_9 + x_grid_var_7 - x_grid_var_9
   - 774 p_right_7_9 <= 0
 non_overlap_down_7_9: h_var_7 + y_grid_var_7 - y_grid_var_9
   + 774 p_down_7_9 <= 774
 non_overlap_down_rev_7_9: - h_var_7 - y_grid_var_7 + y_grid_var_9
   - 774 p_down_7_9 <= 0
 non_overlap_up_7_9: h_var_9 - y_grid_var_7 + y_grid_var_9 + 774 p_up_7_9
   <= 774
 non_overlap_up_rev_7_9: - h_var_9 + y_grid_var_7 - y_grid_var_9
   - 774 p_up_7_9 <= 0
 non_overlap_any_7_10: p_left_7_10 + p_right_7_10 + p_down_7_10 + p_up_7_10
   >= 1
 non_overlap_left_7_10: w_var_7 + x_grid_var_7 - x_grid_var_10
   + 774 p_left_7_10 <= 774
 non_overlap_left_rev_7_10: - w_var_7 - x_grid_var_7 + x_grid_var_10
   - 774 p_left_7_10 <= 0
 non_overlap_right_7_10: w_var_10 - x_grid_var_7 + x_grid_var_10
   + 774 p_right_7_10 <= 774
 non_overlap_right_rev_7_10: - w_var_10 + x_grid_var_7 - x_grid_var_10
   - 774 p_right_7_10 <= 0
 non_overlap_down_7_10: h_var_7 + y_grid_var_7 - y_grid_var_10
   + 774 p_down_7_10 <= 774
 non_overlap_down_rev_7_10: - h_var_7 - y_grid_var_7 + y_grid_var_10
   - 774 p_down_7_10 <= 0
 non_overlap_up_7_10: h_var_10 - y_grid_var_7 + y_grid_var_10
   + 774 p_up_7_10 <= 774
 non_overlap_up_rev_7_10: - h_var_10 + y_grid_var_7 - y_grid_var_10
   - 774 p_up_7_10 <= 0
 non_overlap_any_7_11: p_left_7_11 + p_right_7_11 + p_down_7_11 + p_up_7_11
   >= 1
 non_overlap_left_7_11: w_var_7 + x_grid_var_7 - x_grid_var_11
   + 774 p_left_7_11 <= 774
 non_overlap_left_rev_7_11: - w_var_7 - x_grid_var_7 + x_grid_var_11
   - 774 p_left_7_11 <= 0
 non_overlap_right_7_11: w_var_11 - x_grid_var_7 + x_grid_var_11
   + 774 p_right_7_11 <= 774
 non_overlap_right_rev_7_11: - w_var_11 + x_grid_var_7 - x_grid_var_11
   - 774 p_right_7_11 <= 0
 non_overlap_down_7_11: h_var_7 + y_grid_var_7 - y_grid_var_11
   + 774 p_down_7_11 <= 774
 non_overlap_down_rev_7_11: - h_var_7 - y_grid_var_7 + y_grid_var_11
   - 774 p_down_7_11 <= 0
 non_overlap_up_7_11: h_var_11 - y_grid_var_7 + y_grid_var_11
   + 774 p_up_7_11 <= 774
 non_overlap_up_rev_7_11: - h_var_11 + y_grid_var_7 - y_grid_var_11
   - 774 p_up_7_11 <= 0
 non_overlap_any_7_12: p_left_7_12 + p_right_7_12 + p_down_7_12 + p_up_7_12
   >= 1
 non_overlap_left_7_12: w_var_7 + x_grid_var_7 - x_grid_var_12
   + 774 p_left_7_12 <= 774
 non_overlap_left_rev_7_12: - w_var_7 - x_grid_var_7 + x_grid_var_12
   - 774 p_left_7_12 <= 0
 non_overlap_right_7_12: w_var_12 - x_grid_var_7 + x_grid_var_12
   + 774 p_right_7_12 <= 774
 non_overlap_right_rev_7_12: - w_var_12 + x_grid_var_7 - x_grid_var_12
   - 774 p_right_7_12 <= 0
 non_overlap_down_7_12: h_var_7 + y_grid_var_7 - y_grid_var_12
   + 774 p_down_7_12 <= 774
 non_overlap_down_rev_7_12: - h_var_7 - y_grid_var_7 + y_grid_var_12
   - 774 p_down_7_12 <= 0
 non_overlap_up_7_12: h_var_12 - y_grid_var_7 + y_grid_var_12
   + 774 p_up_7_12 <= 774
 non_overlap_up_rev_7_12: - h_var_12 + y_grid_var_7 - y_grid_var_12
   - 774 p_up_7_12 <= 0
 non_overlap_any_7_13: p_left_7_13 + p_right_7_13 + p_down_7_13 + p_up_7_13
   >= 1
 non_overlap_left_7_13: w_var_7 + x_grid_var_7 - x_grid_var_13
   + 774 p_left_7_13 <= 774
 non_overlap_left_rev_7_13: - w_var_7 - x_grid_var_7 + x_grid_var_13
   - 774 p_left_7_13 <= 0
 non_overlap_right_7_13: w_var_13 - x_grid_var_7 + x_grid_var_13
   + 774 p_right_7_13 <= 774
 non_overlap_right_rev_7_13: - w_var_13 + x_grid_var_7 - x_grid_var_13
   - 774 p_right_7_13 <= 0
 non_overlap_down_7_13: h_var_7 + y_grid_var_7 - y_grid_var_13
   + 774 p_down_7_13 <= 774
 non_overlap_down_rev_7_13: - h_var_7 - y_grid_var_7 + y_grid_var_13
   - 774 p_down_7_13 <= 0
 non_overlap_up_7_13: h_var_13 - y_grid_var_7 + y_grid_var_13
   + 774 p_up_7_13 <= 774
 non_overlap_up_rev_7_13: - h_var_13 + y_grid_var_7 - y_grid_var_13
   - 774 p_up_7_13 <= 0
 non_overlap_any_7_14: p_left_7_14 + p_right_7_14 + p_down_7_14 + p_up_7_14
   >= 1
 non_overlap_left_7_14: w_var_7 + x_grid_var_7 - x_grid_var_14
   + 774 p_left_7_14 <= 774
 non_overlap_left_rev_7_14: - w_var_7 - x_grid_var_7 + x_grid_var_14
   - 774 p_left_7_14 <= 0
 non_overlap_right_7_14: w_var_14 - x_grid_var_7 + x_grid_var_14
   + 774 p_right_7_14 <= 774
 non_overlap_right_rev_7_14: - w_var_14 + x_grid_var_7 - x_grid_var_14
   - 774 p_right_7_14 <= 0
 non_overlap_down_7_14: h_var_7 + y_grid_var_7 - y_grid_var_14
   + 774 p_down_7_14 <= 774
 non_overlap_down_rev_7_14: - h_var_7 - y_grid_var_7 + y_grid_var_14
   - 774 p_down_7_14 <= 0
 non_overlap_up_7_14: h_var_14 - y_grid_var_7 + y_grid_var_14
   + 774 p_up_7_14 <= 774
 non_overlap_up_rev_7_14: - h_var_14 + y_grid_var_7 - y_grid_var_14
   - 774 p_up_7_14 <= 0
 non_overlap_any_7_15: p_left_7_15 + p_right_7_15 + p_down_7_15 + p_up_7_15
   >= 1
 non_overlap_left_7_15: w_var_7 + x_grid_var_7 - x_grid_var_15
   + 774 p_left_7_15 <= 774
 non_overlap_left_rev_7_15: - w_var_7 - x_grid_var_7 + x_grid_var_15
   - 774 p_left_7_15 <= 0
 non_overlap_right_7_15: w_var_15 - x_grid_var_7 + x_grid_var_15
   + 774 p_right_7_15 <= 774
 non_overlap_right_rev_7_15: - w_var_15 + x_grid_var_7 - x_grid_var_15
   - 774 p_right_7_15 <= 0
 non_overlap_down_7_15: h_var_7 + y_grid_var_7 - y_grid_var_15
   + 774 p_down_7_15 <= 774
 non_overlap_down_rev_7_15: - h_var_7 - y_grid_var_7 + y_grid_var_15
   - 774 p_down_7_15 <= 0
 non_overlap_up_7_15: h_var_15 - y_grid_var_7 + y_grid_var_15
   + 774 p_up_7_15 <= 774
 non_overlap_up_rev_7_15: - h_var_15 + y_grid_var_7 - y_grid_var_15
   - 774 p_up_7_15 <= 0
 non_overlap_any_8_9: p_left_8_9 + p_right_8_9 + p_down_8_9 + p_up_8_9
   >= 1
 non_overlap_left_8_9: w_var_8 + x_grid_var_8 - x_grid_var_9
   + 774 p_left_8_9 <= 774
 non_overlap_left_rev_8_9: - w_var_8 - x_grid_var_8 + x_grid_var_9
   - 774 p_left_8_9 <= 0
 non_overlap_right_8_9: w_var_9 - x_grid_var_8 + x_grid_var_9
   + 774 p_right_8_9 <= 774
 non_overlap_right_rev_8_9: - w_var_9 + x_grid_var_8 - x_grid_var_9
   - 774 p_right_8_9 <= 0
 non_overlap_down_8_9: h_var_8 + y_grid_var_8 - y_grid_var_9
   + 774 p_down_8_9 <= 774
 non_overlap_down_rev_8_9: - h_var_8 - y_grid_var_8 + y_grid_var_9
   - 774 p_down_8_9 <= 0
 non_overlap_up_8_9: h_var_9 - y_grid_var_8 + y_grid_var_9 + 774 p_up_8_9
   <= 774
 non_overlap_up_rev_8_9: - h_var_9 + y_grid_var_8 - y_grid_var_9
   - 774 p_up_8_9 <= 0
 non_overlap_any_8_10: p_left_8_10 + p_right_8_10 + p_down_8_10 + p_up_8_10
   >= 1
 non_overlap_left_8_10: w_var_8 + x_grid_var_8 - x_grid_var_10
   + 774 p_left_8_10 <= 774
 non_overlap_left_rev_8_10: - w_var_8 - x_grid_var_8 + x_grid_var_10
   - 774 p_left_8_10 <= 0
 non_overlap_right_8_10: w_var_10 - x_grid_var_8 + x_grid_var_10
   + 774 p_right_8_10 <= 774
 non_overlap_right_rev_8_10: - w_var_10 + x_grid_var_8 - x_grid_var_10
   - 774 p_right_8_10 <= 0
 non_overlap_down_8_10: h_var_8 + y_grid_var_8 - y_grid_var_10
   + 774 p_down_8_10 <= 774
 non_overlap_down_rev_8_10: - h_var_8 - y_grid_var_8 + y_grid_var_10
   - 774 p_down_8_10 <= 0
 non_overlap_up_8_10: h_var_10 - y_grid_var_8 + y_grid_var_10
   + 774 p_up_8_10 <= 774
 non_overlap_up_rev_8_10: - h_var_10 + y_grid_var_8 - y_grid_var_10
   - 774 p_up_8_10 <= 0
 non_overlap_any_8_11: p_left_8_11 + p_right_8_11 + p_down_8_11 + p_up_8_11
   >= 1
 non_overlap_left_8_11: w_var_8 + x_grid_var_8 - x_grid_var_11
   + 774 p_left_8_11 <= 774
 non_overlap_left_rev_8_11: - w_var_8 - x_grid_var_8 + x_grid_var_11
   - 774 p_left_8_11 <= 0
 non_overlap_right_8_11: w_var_11 - x_grid_var_8 + x_grid_var_11
   + 774 p_right_8_11 <= 774
 non_overlap_right_rev_8_11: - w_var_11 + x_grid_var_8 - x_grid_var_11
   - 774 p_right_8_11 <= 0
 non_overlap_down_8_11: h_var_8 + y_grid_var_8 - y_grid_var_11
   + 774 p_down_8_11 <= 774
 non_overlap_down_rev_8_11: - h_var_8 - y_grid_var_8 + y_grid_var_11
   - 774 p_down_8_11 <= 0
 non_overlap_up_8_11: h_var_11 - y_grid_var_8 + y_grid_var_11
   + 774 p_up_8_11 <= 774
 non_overlap_up_rev_8_11: - h_var_11 + y_grid_var_8 - y_grid_var_11
   - 774 p_up_8_11 <= 0
 non_overlap_any_8_12: p_left_8_12 + p_right_8_12 + p_down_8_12 + p_up_8_12
   >= 1
 non_overlap_left_8_12: w_var_8 + x_grid_var_8 - x_grid_var_12
   + 774 p_left_8_12 <= 774
 non_overlap_left_rev_8_12: - w_var_8 - x_grid_var_8 + x_grid_var_12
   - 774 p_left_8_12 <= 0
 non_overlap_right_8_12: w_var_12 - x_grid_var_8 + x_grid_var_12
   + 774 p_right_8_12 <= 774
 non_overlap_right_rev_8_12: - w_var_12 + x_grid_var_8 - x_grid_var_12
   - 774 p_right_8_12 <= 0
 non_overlap_down_8_12: h_var_8 + y_grid_var_8 - y_grid_var_12
   + 774 p_down_8_12 <= 774
 non_overlap_down_rev_8_12: - h_var_8 - y_grid_var_8 + y_grid_var_12
   - 774 p_down_8_12 <= 0
 non_overlap_up_8_12: h_var_12 - y_grid_var_8 + y_grid_var_12
   + 774 p_up_8_12 <= 774
 non_overlap_up_rev_8_12: - h_var_12 + y_grid_var_8 - y_grid_var_12
   - 774 p_up_8_12 <= 0
 non_overlap_any_8_13: p_left_8_13 + p_right_8_13 + p_down_8_13 + p_up_8_13
   >= 1
 non_overlap_left_8_13: w_var_8 + x_grid_var_8 - x_grid_var_13
   + 774 p_left_8_13 <= 774
 non_overlap_left_rev_8_13: - w_var_8 - x_grid_var_8 + x_grid_var_13
   - 774 p_left_8_13 <= 0
 non_overlap_right_8_13: w_var_13 - x_grid_var_8 + x_grid_var_13
   + 774 p_right_8_13 <= 774
 non_overlap_right_rev_8_13: - w_var_13 + x_grid_var_8 - x_grid_var_13
   - 774 p_right_8_13 <= 0
 non_overlap_down_8_13: h_var_8 + y_grid_var_8 - y_grid_var_13
   + 774 p_down_8_13 <= 774
 non_overlap_down_rev_8_13: - h_var_8 - y_grid_var_8 + y_grid_var_13
   - 774 p_down_8_13 <= 0
 non_overlap_up_8_13: h_var_13 - y_grid_var_8 + y_grid_var_13
   + 774 p_up_8_13 <= 774
 non_overlap_up_rev_8_13: - h_var_13 + y_grid_var_8 - y_grid_var_13
   - 774 p_up_8_13 <= 0
 non_overlap_any_8_14: p_left_8_14 + p_right_8_14 + p_down_8_14 + p_up_8_14
   >= 1
 non_overlap_left_8_14: w_var_8 + x_grid_var_8 - x_grid_var_14
   + 774 p_left_8_14 <= 774
 non_overlap_left_rev_8_14: - w_var_8 - x_grid_var_8 + x_grid_var_14
   - 774 p_left_8_14 <= 0
 non_overlap_right_8_14: w_var_14 - x_grid_var_8 + x_grid_var_14
   + 774 p_right_8_14 <= 774
 non_overlap_right_rev_8_14: - w_var_14 + x_grid_var_8 - x_grid_var_14
   - 774 p_right_8_14 <= 0
 non_overlap_down_8_14: h_var_8 + y_grid_var_8 - y_grid_var_14
   + 774 p_down_8_14 <= 774
 non_overlap_down_rev_8_14: - h_var_8 - y_grid_var_8 + y_grid_var_14
   - 774 p_down_8_14 <= 0
 non_overlap_up_8_14: h_var_14 - y_grid_var_8 + y_grid_var_14
   + 774 p_up_8_14 <= 774
 non_overlap_up_rev_8_14: - h_var_14 + y_grid_var_8 - y_grid_var_14
   - 774 p_up_8_14 <= 0
 non_overlap_any_8_15: p_left_8_15 + p_right_8_15 + p_down_8_15 + p_up_8_15
   >= 1
 non_overlap_left_8_15: w_var_8 + x_grid_var_8 - x_grid_var_15
   + 774 p_left_8_15 <= 774
 non_overlap_left_rev_8_15: - w_var_8 - x_grid_var_8 + x_grid_var_15
   - 774 p_left_8_15 <= 0
 non_overlap_right_8_15: w_var_15 - x_grid_var_8 + x_grid_var_15
   + 774 p_right_8_15 <= 774
 non_overlap_right_rev_8_15: - w_var_15 + x_grid_var_8 - x_grid_var_15
   - 774 p_right_8_15 <= 0
 non_overlap_down_8_15: h_var_8 + y_grid_var_8 - y_grid_var_15
   + 774 p_down_8_15 <= 774
 non_overlap_down_rev_8_15: - h_var_8 - y_grid_var_8 + y_grid_var_15
   - 774 p_down_8_15 <= 0
 non_overlap_up_8_15: h_var_15 - y_grid_var_8 + y_grid_var_15
   + 774 p_up_8_15 <= 774
 non_overlap_up_rev_8_15: - h_var_15 + y_grid_var_8 - y_grid_var_15
   - 774 p_up_8_15 <= 0
 non_overlap_any_9_10: p_left_9_10 + p_right_9_10 + p_down_9_10 + p_up_9_10
   >= 1
 non_overlap_left_9_10: w_var_9 + x_grid_var_9 - x_grid_var_10
   + 774 p_left_9_10 <= 774
 non_overlap_left_rev_9_10: - w_var_9 - x_grid_var_9 + x_grid_var_10
   - 774 p_left_9_10 <= 0
 non_overlap_right_9_10: w_var_10 - x_grid_var_9 + x_grid_var_10
   + 774 p_right_9_10 <= 774
 non_overlap_right_rev_9_10: - w_var_10 + x_grid_var_9 - x_grid_var_10
   - 774 p_right_9_10 <= 0
 non_overlap_down_9_10: h_var_9 + y_grid_var_9 - y_grid_var_10
   + 774 p_down_9_10 <= 774
 non_overlap_down_rev_9_10: - h_var_9 - y_grid_var_9 + y_grid_var_10
   - 774 p_down_9_10 <= 0
 non_overlap_up_9_10: h_var_10 - y_grid_var_9 + y_grid_var_10
   + 774 p_up_9_10 <= 774
 non_overlap_up_rev_9_10: - h_var_10 + y_grid_var_9 - y_grid_var_10
   - 774 p_up_9_10 <= 0
 non_overlap_any_9_11: p_left_9_11 + p_right_9_11 + p_down_9_11 + p_up_9_11
   >= 1
 non_overlap_left_9_11: w_var_9 + x_grid_var_9 - x_grid_var_11
   + 774 p_left_9_11 <= 774
 non_overlap_left_rev_9_11: - w_var_9 - x_grid_var_9 + x_grid_var_11
   - 774 p_left_9_11 <= 0
 non_overlap_right_9_11: w_var_11 - x_grid_var_9 + x_grid_var_11
   + 774 p_right_9_11 <= 774
 non_overlap_right_rev_9_11: - w_var_11 + x_grid_var_9 - x_grid_var_11
   - 774 p_right_9_11 <= 0
 non_overlap_down_9_11: h_var_9 + y_grid_var_9 - y_grid_var_11
   + 774 p_down_9_11 <= 774
 non_overlap_down_rev_9_11: - h_var_9 - y_grid_var_9 + y_grid_var_11
   - 774 p_down_9_11 <= 0
 non_overlap_up_9_11: h_var_11 - y_grid_var_9 + y_grid_var_11
   + 774 p_up_9_11 <= 774
 non_overlap_up_rev_9_11: - h_var_11 + y_grid_var_9 - y_grid_var_11
   - 774 p_up_9_11 <= 0
 non_overlap_any_9_12: p_left_9_12 + p_right_9_12 + p_down_9_12 + p_up_9_12
   >= 1
 non_overlap_left_9_12: w_var_9 + x_grid_var_9 - x_grid_var_12
   + 774 p_left_9_12 <= 774
 non_overlap_left_rev_9_12: - w_var_9 - x_grid_var_9 + x_grid_var_12
   - 774 p_left_9_12 <= 0
 non_overlap_right_9_12: w_var_12 - x_grid_var_9 + x_grid_var_12
   + 774 p_right_9_12 <= 774
 non_overlap_right_rev_9_12: - w_var_12 + x_grid_var_9 - x_grid_var_12
   - 774 p_right_9_12 <= 0
 non_overlap_down_9_12: h_var_9 + y_grid_var_9 - y_grid_var_12
   + 774 p_down_9_12 <= 774
 non_overlap_down_rev_9_12: - h_var_9 - y_grid_var_9 + y_grid_var_12
   - 774 p_down_9_12 <= 0
 non_overlap_up_9_12: h_var_12 - y_grid_var_9 + y_grid_var_12
   + 774 p_up_9_12 <= 774
 non_overlap_up_rev_9_12: - h_var_12 + y_grid_var_9 - y_grid_var_12
   - 774 p_up_9_12 <= 0
 non_overlap_any_9_13: p_left_9_13 + p_right_9_13 + p_down_9_13 + p_up_9_13
   >= 1
 non_overlap_left_9_13: w_var_9 + x_grid_var_9 - x_grid_var_13
   + 774 p_left_9_13 <= 774
 non_overlap_left_rev_9_13: - w_var_9 - x_grid_var_9 + x_grid_var_13
   - 774 p_left_9_13 <= 0
 non_overlap_right_9_13: w_var_13 - x_grid_var_9 + x_grid_var_13
   + 774 p_right_9_13 <= 774
 non_overlap_right_rev_9_13: - w_var_13 + x_grid_var_9 - x_grid_var_13
   - 774 p_right_9_13 <= 0
 non_overlap_down_9_13: h_var_9 + y_grid_var_9 - y_grid_var_13
   + 774 p_down_9_13 <= 774
 non_overlap_down_rev_9_13: - h_var_9 - y_grid_var_9 + y_grid_var_13
   - 774 p_down_9_13 <= 0
 non_overlap_up_9_13: h_var_13 - y_grid_var_9 + y_grid_var_13
   + 774 p_up_9_13 <= 774
 non_overlap_up_rev_9_13: - h_var_13 + y_grid_var_9 - y_grid_var_13
   - 774 p_up_9_13 <= 0
 non_overlap_any_9_14: p_left_9_14 + p_right_9_14 + p_down_9_14 + p_up_9_14
   >= 1
 non_overlap_left_9_14: w_var_9 + x_grid_var_9 - x_grid_var_14
   + 774 p_left_9_14 <= 774
 non_overlap_left_rev_9_14: - w_var_9 - x_grid_var_9 + x_grid_var_14
   - 774 p_left_9_14 <= 0
 non_overlap_right_9_14: w_var_14 - x_grid_var_9 + x_grid_var_14
   + 774 p_right_9_14 <= 774
 non_overlap_right_rev_9_14: - w_var_14 + x_grid_var_9 - x_grid_var_14
   - 774 p_right_9_14 <= 0
 non_overlap_down_9_14: h_var_9 + y_grid_var_9 - y_grid_var_14
   + 774 p_down_9_14 <= 774
 non_overlap_down_rev_9_14: - h_var_9 - y_grid_var_9 + y_grid_var_14
   - 774 p_down_9_14 <= 0
 non_overlap_up_9_14: h_var_14 - y_grid_var_9 + y_grid_var_14
   + 774 p_up_9_14 <= 774
 non_overlap_up_rev_9_14: - h_var_14 + y_grid_var_9 - y_grid_var_14
   - 774 p_up_9_14 <= 0
 non_overlap_any_9_15: p_left_9_15 + p_right_9_15 + p_down_9_15 + p_up_9_15
   >= 1
 non_overlap_left_9_15: w_var_9 + x_grid_var_9 - x_grid_var_15
   + 774 p_left_9_15 <= 774
 non_overlap_left_rev_9_15: - w_var_9 - x_grid_var_9 + x_grid_var_15
   - 774 p_left_9_15 <= 0
 non_overlap_right_9_15: w_var_15 - x_grid_var_9 + x_grid_var_15
   + 774 p_right_9_15 <= 774
 non_overlap_right_rev_9_15: - w_var_15 + x_grid_var_9 - x_grid_var_15
   - 774 p_right_9_15 <= 0
 non_overlap_down_9_15: h_var_9 + y_grid_var_9 - y_grid_var_15
   + 774 p_down_9_15 <= 774
 non_overlap_down_rev_9_15: - h_var_9 - y_grid_var_9 + y_grid_var_15
   - 774 p_down_9_15 <= 0
 non_overlap_up_9_15: h_var_15 - y_grid_var_9 + y_grid_var_15
   + 774 p_up_9_15 <= 774
 non_overlap_up_rev_9_15: - h_var_15 + y_grid_var_9 - y_grid_var_15
   - 774 p_up_9_15 <= 0
 non_overlap_any_10_11: p_left_10_11 + p_right_10_11 + p_down_10_11
   + p_up_10_11 >= 1
 non_overlap_left_10_11: w_var_10 + x_grid_var_10 - x_grid_var_11
   + 774 p_left_10_11 <= 774
 non_overlap_left_rev_10_11: - w_var_10 - x_grid_var_10 + x_grid_var_11
   - 774 p_left_10_11 <= 0
 non_overlap_right_10_11: w_var_11 - x_grid_var_10 + x_grid_var_11
   + 774 p_right_10_11 <= 774
 non_overlap_right_rev_10_11: - w_var_11 + x_grid_var_10 - x_grid_var_11
   - 774 p_right_10_11 <= 0
 non_overlap_down_10_11: h_var_10 + y_grid_var_10 - y_grid_var_11
   + 774 p_down_10_11 <= 774
 non_overlap_down_rev_10_11: - h_var_10 - y_grid_var_10 + y_grid_var_11
   - 774 p_down_10_11 <= 0
 non_overlap_up_10_11: h_var_11 - y_grid_var_10 + y_grid_var_11
   + 774 p_up_10_11 <= 774
 non_overlap_up_rev_10_11: - h_var_11 + y_grid_var_10 - y_grid_var_11
   - 774 p_up_10_11 <= 0
 non_overlap_any_10_12: p_left_10_12 + p_right_10_12 + p_down_10_12
   + p_up_10_12 >= 1
 non_overlap_left_10_12: w_var_10 + x_grid_var_10 - x_grid_var_12
   + 774 p_left_10_12 <= 774
 non_overlap_left_rev_10_12: - w_var_10 - x_grid_var_10 + x_grid_var_12
   - 774 p_left_10_12 <= 0
 non_overlap_right_10_12: w_var_12 - x_grid_var_10 + x_grid_var_12
   + 774 p_right_10_12 <= 774
 non_overlap_right_rev_10_12: - w_var_12 + x_grid_var_10 - x_grid_var_12
   - 774 p_right_10_12 <= 0
 non_overlap_down_10_12: h_var_10 + y_grid_var_10 - y_grid_var_12
   + 774 p_down_10_12 <= 774
 non_overlap_down_rev_10_12: - h_var_10 - y_grid_var_10 + y_grid_var_12
   - 774 p_down_10_12 <= 0
 non_overlap_up_10_12: h_var_12 - y_grid_var_10 + y_grid_var_12
   + 774 p_up_10_12 <= 774
 non_overlap_up_rev_10_12: - h_var_12 + y_grid_var_10 - y_grid_var_12
   - 774 p_up_10_12 <= 0
 non_overlap_any_10_13: p_left_10_13 + p_right_10_13 + p_down_10_13
   + p_up_10_13 >= 1
 non_overlap_left_10_13: w_var_10 + x_grid_var_10 - x_grid_var_13
   + 774 p_left_10_13 <= 774
 non_overlap_left_rev_10_13: - w_var_10 - x_grid_var_10 + x_grid_var_13
   - 774 p_left_10_13 <= 0
 non_overlap_right_10_13: w_var_13 - x_grid_var_10 + x_grid_var_13
   + 774 p_right_10_13 <= 774
 non_overlap_right_rev_10_13: - w_var_13 + x_grid_var_10 - x_grid_var_13
   - 774 p_right_10_13 <= 0
 non_overlap_down_10_13: h_var_10 + y_grid_var_10 - y_grid_var_13
   + 774 p_down_10_13 <= 774
 non_overlap_down_rev_10_13: - h_var_10 - y_grid_var_10 + y_grid_var_13
   - 774 p_down_10_13 <= 0
 non_overlap_up_10_13: h_var_13 - y_grid_var_10 + y_grid_var_13
   + 774 p_up_10_13 <= 774
 non_overlap_up_rev_10_13: - h_var_13 + y_grid_var_10 - y_grid_var_13
   - 774 p_up_10_13 <= 0
 non_overlap_any_10_14: p_left_10_14 + p_right_10_14 + p_down_10_14
   + p_up_10_14 >= 1
 non_overlap_left_10_14: w_var_10 + x_grid_var_10 - x_grid_var_14
   + 774 p_left_10_14 <= 774
 non_overlap_left_rev_10_14: - w_var_10 - x_grid_var_10 + x_grid_var_14
   - 774 p_left_10_14 <= 0
 non_overlap_right_10_14: w_var_14 - x_grid_var_10 + x_grid_var_14
   + 774 p_right_10_14 <= 774
 non_overlap_right_rev_10_14: - w_var_14 + x_grid_var_10 - x_grid_var_14
   - 774 p_right_10_14 <= 0
 non_overlap_down_10_14: h_var_10 + y_grid_var_10 - y_grid_var_14
   + 774 p_down_10_14 <= 774
 non_overlap_down_rev_10_14: - h_var_10 - y_grid_var_10 + y_grid_var_14
   - 774 p_down_10_14 <= 0
 non_overlap_up_10_14: h_var_14 - y_grid_var_10 + y_grid_var_14
   + 774 p_up_10_14 <= 774
 non_overlap_up_rev_10_14: - h_var_14 + y_grid_var_10 - y_grid_var_14
   - 774 p_up_10_14 <= 0
 non_overlap_any_10_15: p_left_10_15 + p_right_10_15 + p_down_10_15
   + p_up_10_15 >= 1
 non_overlap_left_10_15: w_var_10 + x_grid_var_10 - x_grid_var_15
   + 774 p_left_10_15 <= 774
 non_overlap_left_rev_10_15: - w_var_10 - x_grid_var_10 + x_grid_var_15
   - 774 p_left_10_15 <= 0
 non_overlap_right_10_15: w_var_15 - x_grid_var_10 + x_grid_var_15
   + 774 p_right_10_15 <= 774
 non_overlap_right_rev_10_15: - w_var_15 + x_grid_var_10 - x_grid_var_15
   - 774 p_right_10_15 <= 0
 non_overlap_down_10_15: h_var_10 + y_grid_var_10 - y_grid_var_15
   + 774 p_down_10_15 <= 774
 non_overlap_down_rev_10_15: - h_var_10 - y_grid_var_10 + y_grid_var_15
   - 774 p_down_10_15 <= 0
 non_overlap_up_10_15: h_var_15 - y_grid_var_10 + y_grid_var_15
   + 774 p_up_10_15 <= 774
 non_overlap_up_rev_10_15: - h_var_15 + y_grid_var_10 - y_grid_var_15
   - 774 p_up_10_15 <= 0
 non_overlap_any_11_12: p_left_11_12 + p_right_11_12 + p_down_11_12
   + p_up_11_12 >= 1
 non_overlap_left_11_12: w_var_11 + x_grid_var_11 - x_grid_var_12
   + 774 p_left_11_12 <= 774
 non_overlap_left_rev_11_12: - w_var_11 - x_grid_var_11 + x_grid_var_12
   - 774 p_left_11_12 <= 0
 non_overlap_right_11_12: w_var_12 - x_grid_var_11 + x_grid_var_12
   + 774 p_right_11_12 <= 774
 non_overlap_right_rev_11_12: - w_var_12 + x_grid_var_11 - x_grid_var_12
   - 774 p_right_11_12 <= 0
 non_overlap_down_11_12: h_var_11 + y_grid_var_11 - y_grid_var_12
   + 774 p_down_11_12 <= 774
 non_overlap_down_rev_11_12: - h_var_11 - y_grid_var_11 + y_grid_var_12
   - 774 p_down_11_12 <= 0
 non_overlap_up_11_12: h_var_12 - y_grid_var_11 + y_grid_var_12
   + 774 p_up_11_12 <= 774
 non_overlap_up_rev_11_12: - h_var_12 + y_grid_var_11 - y_grid_var_12
   - 774 p_up_11_12 <= 0
 non_overlap_any_11_13: p_left_11_13 + p_right_11_13 + p_down_11_13
   + p_up_11_13 >= 1
 non_overlap_left_11_13: w_var_11 + x_grid_var_11 - x_grid_var_13
   + 774 p_left_11_13 <= 774
 non_overlap_left_rev_11_13: - w_var_11 - x_grid_var_11 + x_grid_var_13
   - 774 p_left_11_13 <= 0
 non_overlap_right_11_13: w_var_13 - x_grid_var_11 + x_grid_var_13
   + 774 p_right_11_13 <= 774
 non_overlap_right_rev_11_13: - w_var_13 + x_grid_var_11 - x_grid_var_13
   - 774 p_right_11_13 <= 0
 non_overlap_down_11_13: h_var_11 + y_grid_var_11 - y_grid_var_13
   + 774 p_down_11_13 <= 774
 non_overlap_down_rev_11_13: - h_var_11 - y_grid_var_11 + y_grid_var_13
   - 774 p_down_11_13 <= 0
 non_overlap_up_11_13: h_var_13 - y_grid_var_11 + y_grid_var_13
   + 774 p_up_11_13 <= 774
 non_overlap_up_rev_11_13: - h_var_13 + y_grid_var_11 - y_grid_var_13
   - 774 p_up_11_13 <= 0
 non_overlap_any_11_14: p_left_11_14 + p_right_11_14 + p_down_11_14
   + p_up_11_14 >= 1
 non_overlap_left_11_14: w_var_11 + x_grid_var_11 - x_grid_var_14
   + 774 p_left_11_14 <= 774
 non_overlap_left_rev_11_14: - w_var_11 - x_grid_var_11 + x_grid_var_14
   - 774 p_left_11_14 <= 0
 non_overlap_right_11_14: w_var_14 - x_grid_var_11 + x_grid_var_14
   + 774 p_right_11_14 <= 774
 non_overlap_right_rev_11_14: - w_var_14 + x_grid_var_11 - x_grid_var_14
   - 774 p_right_11_14 <= 0
 non_overlap_down_11_14: h_var_11 + y_grid_var_11 - y_grid_var_14
   + 774 p_down_11_14 <= 774
 non_overlap_down_rev_11_14: - h_var_11 - y_grid_var_11 + y_grid_var_14
   - 774 p_down_11_14 <= 0
 non_overlap_up_11_14: h_var_14 - y_grid_var_11 + y_grid_var_14
   + 774 p_up_11_14 <= 774
 non_overlap_up_rev_11_14: - h_var_14 + y_grid_var_11 - y_grid_var_14
   - 774 p_up_11_14 <= 0
 non_overlap_any_11_15: p_left_11_15 + p_right_11_15 + p_down_11_15
   + p_up_11_15 >= 1
 non_overlap_left_11_15: w_var_11 + x_grid_var_11 - x_grid_var_15
   + 774 p_left_11_15 <= 774
 non_overlap_left_rev_11_15: - w_var_11 - x_grid_var_11 + x_grid_var_15
   - 774 p_left_11_15 <= 0
 non_overlap_right_11_15: w_var_15 - x_grid_var_11 + x_grid_var_15
   + 774 p_right_11_15 <= 774
 non_overlap_right_rev_11_15: - w_var_15 + x_grid_var_11 - x_grid_var_15
   - 774 p_right_11_15 <= 0
 non_overlap_down_11_15: h_var_11 + y_grid_var_11 - y_grid_var_15
   + 774 p_down_11_15 <= 774
 non_overlap_down_rev_11_15: - h_var_11 - y_grid_var_11 + y_grid_var_15
   - 774 p_down_11_15 <= 0
 non_overlap_up_11_15: h_var_15 - y_grid_var_11 + y_grid_var_15
   + 774 p_up_11_15 <= 774
 non_overlap_up_rev_11_15: - h_var_15 + y_grid_var_11 - y_grid_var_15
   - 774 p_up_11_15 <= 0
 non_overlap_any_12_13: p_left_12_13 + p_right_12_13 + p_down_12_13
   + p_up_12_13 >= 1
 non_overlap_left_12_13: w_var_12 + x_grid_var_12 - x_grid_var_13
   + 774 p_left_12_13 <= 774
 non_overlap_left_rev_12_13: - w_var_12 - x_grid_var_12 + x_grid_var_13
   - 774 p_left_12_13 <= 0
 non_overlap_right_12_13: w_var_13 - x_grid_var_12 + x_grid_var_13
   + 774 p_right_12_13 <= 774
 non_overlap_right_rev_12_13: - w_var_13 + x_grid_var_12 - x_grid_var_13
   - 774 p_right_12_13 <= 0
 non_overlap_down_12_13: h_var_12 + y_grid_var_12 - y_grid_var_13
   + 774 p_down_12_13 <= 774
 non_overlap_down_rev_12_13: - h_var_12 - y_grid_var_12 + y_grid_var_13
   - 774 p_down_12_13 <= 0
 non_overlap_up_12_13: h_var_13 - y_grid_var_12 + y_grid_var_13
   + 774 p_up_12_13 <= 774
 non_overlap_up_rev_12_13: - h_var_13 + y_grid_var_12 - y_grid_var_13
   - 774 p_up_12_13 <= 0
 non_overlap_any_12_14: p_left_12_14 + p_right_12_14 + p_down_12_14
   + p_up_12_14 >= 1
 non_overlap_left_12_14: w_var_12 + x_grid_var_12 - x_grid_var_14
   + 774 p_left_12_14 <= 774
 non_overlap_left_rev_12_14: - w_var_12 - x_grid_var_12 + x_grid_var_14
   - 774 p_left_12_14 <= 0
 non_overlap_right_12_14: w_var_14 - x_grid_var_12 + x_grid_var_14
   + 774 p_right_12_14 <= 774
 non_overlap_right_rev_12_14: - w_var_14 + x_grid_var_12 - x_grid_var_14
   - 774 p_right_12_14 <= 0
 non_overlap_down_12_14: h_var_12 + y_grid_var_12 - y_grid_var_14
   + 774 p_down_12_14 <= 774
 non_overlap_down_rev_12_14: - h_var_12 - y_grid_var_12 + y_grid_var_14
   - 774 p_down_12_14 <= 0
 non_overlap_up_12_14: h_var_14 - y_grid_var_12 + y_grid_var_14
   + 774 p_up_12_14 <= 774
 non_overlap_up_rev_12_14: - h_var_14 + y_grid_var_12 - y_grid_var_14
   - 774 p_up_12_14 <= 0
 non_overlap_any_12_15: p_left_12_15 + p_right_12_15 + p_down_12_15
   + p_up_12_15 >= 1
 non_overlap_left_12_15: w_var_12 + x_grid_var_12 - x_grid_var_15
   + 774 p_left_12_15 <= 774
 non_overlap_left_rev_12_15: - w_var_12 - x_grid_var_12 + x_grid_var_15
   - 774 p_left_12_15 <= 0
 non_overlap_right_12_15: w_var_15 - x_grid_var_12 + x_grid_var_15
   + 774 p_right_12_15 <= 774
 non_overlap_right_rev_12_15: - w_var_15 + x_grid_var_12 - x_grid_var_15
   - 774 p_right_12_15 <= 0
 non_overlap_down_12_15: h_var_12 + y_grid_var_12 - y_grid_var_15
   + 774 p_down_12_15 <= 774
 non_overlap_down_rev_12_15: - h_var_12 - y_grid_var_12 + y_grid_var_15
   - 774 p_down_12_15 <= 0
 non_overlap_up_12_15: h_var_15 - y_grid_var_12 + y_grid_var_15
   + 774 p_up_12_15 <= 774
 non_overlap_up_rev_12_15: - h_var_15 + y_grid_var_12 - y_grid_var_15
   - 774 p_up_12_15 <= 0
 non_overlap_any_13_14: p_left_13_14 + p_right_13_14 + p_down_13_14
   + p_up_13_14 >= 1
 non_overlap_left_13_14: w_var_13 + x_grid_var_13 - x_grid_var_14
   + 774 p_left_13_14 <= 774
 non_overlap_left_rev_13_14: - w_var_13 - x_grid_var_13 + x_grid_var_14
   - 774 p_left_13_14 <= 0
 non_overlap_right_13_14: w_var_14 - x_grid_var_13 + x_grid_var_14
   + 774 p_right_13_14 <= 774
 non_overlap_right_rev_13_14: - w_var_14 + x_grid_var_13 - x_grid_var_14
   - 774 p_right_13_14 <= 0
 non_overlap_down_13_14: h_var_13 + y_grid_var_13 - y_grid_var_14
   + 774 p_down_13_14 <= 774
 non_overlap_down_rev_13_14: - h_var_13 - y_grid_var_13 + y_grid_var_14
   - 774 p_down_13_14 <= 0
 non_overlap_up_13_14: h_var_14 - y_grid_var_13 + y_grid_var_14
   + 774 p_up_13_14 <= 774
 non_overlap_up_rev_13_14: - h_var_14 + y_grid_var_13 - y_grid_var_14
   - 774 p_up_13_14 <= 0
 non_overlap_any_13_15: p_left_13_15 + p_right_13_15 + p_down_13_15
   + p_up_13_15 >= 1
 non_overlap_left_13_15: w_var_13 + x_grid_var_13 - x_grid_var_15
   + 774 p_left_13_15 <= 774
 non_overlap_left_rev_13_15: - w_var_13 - x_grid_var_13 + x_grid_var_15
   - 774 p_left_13_15 <= 0
 non_overlap_right_13_15: w_var_15 - x_grid_var_13 + x_grid_var_15
   + 774 p_right_13_15 <= 774
 non_overlap_right_rev_13_15: - w_var_15 + x_grid_var_13 - x_grid_var_15
   - 774 p_right_13_15 <= 0
 non_overlap_down_13_15: h_var_13 + y_grid_var_13 - y_grid_var_15
   + 774 p_down_13_15 <= 774
 non_overlap_down_rev_13_15: - h_var_13 - y_grid_var_13 + y_grid_var_15
   - 774 p_down_13_15 <= 0
 non_overlap_up_13_15: h_var_15 - y_grid_var_13 + y_grid_var_15
   + 774 p_up_13_15 <= 774
 non_overlap_up_rev_13_15: - h_var_15 + y_grid_var_13 - y_grid_var_15
   - 774 p_up_13_15 <= 0
 non_overlap_any_14_15: p_left_14_15 + p_right_14_15 + p_down_14_15
   + p_up_14_15 >= 1
 non_overlap_left_14_15: w_var_14 + x_grid_var_14 - x_grid_var_15
   + 774 p_left_14_15 <= 774
 non_overlap_left_rev_14_15: - w_var_14 - x_grid_var_14 + x_grid_var_15
   - 774 p_left_14_15 <= 0
 non_overlap_right_14_15: w_var_15 - x_grid_var_14 + x_grid_var_15
   + 774 p_right_14_15 <= 774
 non_overlap_right_rev_14_15: - w_var_15 + x_grid_var_14 - x_grid_var_15
   - 774 p_right_14_15 <= 0
 non_overlap_down_14_15: h_var_14 + y_grid_var_14 - y_grid_var_15
   + 774 p_down_14_15 <= 774
 non_overlap_down_rev_14_15: - h_var_14 - y_grid_var_14 + y_grid_var_15
   - 774 p_down_14_15 <= 0
 non_overlap_up_14_15: h_var_15 - y_grid_var_14 + y_grid_var_15
   + 774 p_up_14_15 <= 774
 non_overlap_up_rev_14_15: - h_var_15 + y_grid_var_14 - y_grid_var_15
   - 774 p_up_14_15 <= 0
 bbox_min_x_0: - x_grid_var_0 + bbox_min_x <= 0
 bbox_max_x_0: - w_var_0 - x_grid_var_0 + bbox_max_x >= 0
 bbox_min_y_0: - y_grid_var_0 + bbox_min_y <= 0
 bbox_max_y_0: - h_var_0 - y_grid_var_0 + bbox_max_y >= 0
 bbox_min_x_1: - x_grid_var_1 + bbox_min_x <= 0
 bbox_max_x_1: - w_var_1 - x_grid_var_1 + bbox_max_x >= 0
 bbox_min_y_1: - y_grid_var_1 + bbox_min_y <= 0
 bbox_max_y_1: - h_var_1 - y_grid_var_1 + bbox_max_y >= 0
 bbox_min_x_2: - x_grid_var_2 + bbox_min_x <= 0
 bbox_max_x_2: - w_var_2 - x_grid_var_2 + bbox_max_x >= 0
 bbox_min_y_2: - y_grid_var_2 + bbox_min_y <= 0
 bbox_max_y_2: - h_var_2 - y_grid_var_2 + bbox_max_y >= 0
 bbox_min_x_3: - x_grid_var_3 + bbox_min_x <= 0
 bbox_max_x_3: - w_var_3 - x_grid_var_3 + bbox_max_x >= 0
 bbox_min_y_3: - y_grid_var_3 + bbox_min_y <= 0
 bbox_max_y_3: - h_var_3 - y_grid_var_3 + bbox_max_y >= 0
 bbox_min_x_4: - x_grid_var_4 + bbox_min_x <= 0
 bbox_max_x_4: - w_var_4 - x_grid_var_4 + bbox_max_x >= 0
 bbox_min_y_4: - y_grid_var_4 + bbox_min_y <= 0
 bbox_max_y_4: - h_var_4 - y_grid_var_4 + bbox_max_y >= 0
 bbox_min_x_5: - x_grid_var_5 + bbox_min_x <= 0
 bbox_max_x_5: - w_var_5 - x_grid_var_5 + bbox_max_x >= 0
 bbox_min_y_5: - y_grid_var_5 + bbox_min_y <= 0
 bbox_max_y_5: - h_var_5 - y_grid_var_5 + bbox_max_y >= 0
 bbox_min_x_6: - x_grid_var_6 + bbox_min_x <= 0
 bbox_max_x_6: - w_var_6 - x_grid_var_6 + bbox_max_x >= 0
 bbox_min_y_6: - y_grid_var_6 + bbox_min_y <= 0
 bbox_max_y_6: - h_var_6 - y_grid_var_6 + bbox_max_y >= 0
 bbox_min_x_7: - x_grid_var_7 + bbox_min_x <= 0
 bbox_max_x_7: - w_var_7 - x_grid_var_7 + bbox_max_x >= 0
 bbox_min_y_7: - y_grid_var_7 + bbox_min_y <= 0
 bbox_max_y_7: - h_var_7 - y_grid_var_7 + bbox_max_y >= 0
 bbox_min_x_8: - x_grid_var_8 + bbox_min_x <= 0
 bbox_max_x_8: - w_var_8 - x_grid_var_8 + bbox_max_x >= 0
 bbox_min_y_8: - y_grid_var_8 + bbox_min_y <= 0
 bbox_max_y_8: - h_var_8 - y_grid_var_8 + bbox_max_y >= 0
 bbox_min_x_9: - x_grid_var_9 + bbox_min_x <= 0
 bbox_max_x_9: - w_var_9 - x_grid_var_9 + bbox_max_x >= 0
 bbox_min_y_9: - y_grid_var_9 + bbox_min_y <= 0
 bbox_max_y_9: - h_var_9 - y_grid_var_9 + bbox_max_y >= 0
 bbox_min_x_10: - x_grid_var_10 + bbox_min_x <= 0
 bbox_max_x_10: - w_var_10 - x_grid_var_10 + bbox_max_x >= 0
 bbox_min_y_10: - y_grid_var_10 + bbox_min_y <= 0
 bbox_max_y_10: - h_var_10 - y_grid_var_10 + bbox_max_y >= 0
 bbox_min_x_11: - x_grid_var_11 + bbox_min_x <= 0
 bbox_max_x_11: - w_var_11 - x_grid_var_11 + bbox_max_x >= 0
 bbox_min_y_11: - y_grid_var_11 + bbox_min_y <= 0
 bbox_max_y_11: - h_var_11 - y_grid_var_11 + bbox_max_y >= 0
 bbox_min_x_12: - x_grid_var_12 + bbox_min_x <= 0
 bbox_max_x_12: - w_var_12 - x_grid_var_12 + bbox_max_x >= 0
 bbox_min_y_12: - y_grid_var_12 + bbox_min_y <= 0
 bbox_max_y_12: - h_var_12 - y_grid_var_12 + bbox_max_y >= 0
 bbox_min_x_13: - x_grid_var_13 + bbox_min_x <= 0
 bbox_max_x_13: - w_var_13 - x_grid_var_13 + bbox_max_x >= 0
 bbox_min_y_13: - y_grid_var_13 + bbox_min_y <= 0
 bbox_max_y_13: - h_var_13 - y_grid_var_13 + bbox_max_y >= 0
 bbox_min_x_14: - x_grid_var_14 + bbox_min_x <= 0
 bbox_max_x_14: - w_var_14 - x_grid_var_14 + bbox_max_x >= 0
 bbox_min_y_14: - y_grid_var_14 + bbox_min_y <= 0
 bbox_max_y_14: - h_var_14 - y_grid_var_14 + bbox_max_y >= 0
 bbox_min_x_15: - x_grid_var_15 + bbox_min_x <= 0
 bbox_max_x_15: - w_var_15 - x_grid_var_15 + bbox_max_x >= 0
 bbox_min_y_15: - y_grid_var_15 + bbox_min_y <= 0
 bbox_max_y_15: - h_var_15 - y_grid_var_15 + bbox_max_y >= 0
 bbox_w_def: bbox_min_x - bbox_max_x + bbox_w = 0
 bbox_h_def: bbox_min_y - bbox_max_y + bbox_h = 0
 cx_center_def: cx_center - 0.5 bbox_min_x - 0.5 bbox_max_x = 0
 cy_center_def: cy_center - 0.5 bbox_min_y - 0.5 bbox_max_y = 0
 aspect_ratio_diff_ge_w_minus_h: - bbox_w + bbox_h + aspect_ratio_penalty
   >= 0
 aspect_ratio_diff_ge_h_minus_w: bbox_w - bbox_h + aspect_ratio_penalty
   >= 0
 dx_diff_def_0_2: - cx_grid_var_0 + cx_grid_var_2 + dx_diff_0_2 = 0
 dy_diff_def_0_2: - cy_grid_var_0 + cy_grid_var_2 + dy_diff_0_2 = 0
 dx_abs_0_2_abs_ge_orig: dx_abs_0_2 - dx_diff_0_2
   - 774 dx_abs_0_2_is_positive >= -774
 dx_abs_0_2_abs_le_orig: dx_abs_0_2 - dx_diff_0_2
   + 774 dx_abs_0_2_is_positive <= 774
 dx_abs_0_2_abs_ge_neg_orig: dx_abs_0_2 + dx_diff_0_2
   + 774 dx_abs_0_2_is_positive >= 0
 dx_abs_0_2_abs_le_neg_orig: dx_abs_0_2 + dx_diff_0_2
   - 774 dx_abs_0_2_is_positive <= 0
 dx_abs_0_2_force_positive: dx_diff_0_2 - 774 dx_abs_0_2_is_positive
   >= -774
 dx_abs_0_2_force_negative: dx_diff_0_2 - 774 dx_abs_0_2_is_positive <= 0
 dy_abs_0_2_abs_ge_orig: dy_abs_0_2 - dy_diff_0_2
   - 774 dy_abs_0_2_is_positive >= -774
 dy_abs_0_2_abs_le_orig: dy_abs_0_2 - dy_diff_0_2
   + 774 dy_abs_0_2_is_positive <= 774
 dy_abs_0_2_abs_ge_neg_orig: dy_abs_0_2 + dy_diff_0_2
   + 774 dy_abs_0_2_is_positive >= 0
 dy_abs_0_2_abs_le_neg_orig: dy_abs_0_2 + dy_diff_0_2
   - 774 dy_abs_0_2_is_positive <= 0
 dy_abs_0_2_force_positive: dy_diff_0_2 - 774 dy_abs_0_2_is_positive
   >= -774
 dy_abs_0_2_force_negative: dy_diff_0_2 - 774 dy_abs_0_2_is_positive <= 0
 dx_diff_def_0_3: - cx_grid_var_0 + cx_grid_var_3 + dx_diff_0_3 = 0
 dy_diff_def_0_3: - cy_grid_var_0 + cy_grid_var_3 + dy_diff_0_3 = 0
 dx_abs_0_3_abs_ge_orig: dx_abs_0_3 - dx_diff_0_3
   - 774 dx_abs_0_3_is_positive >= -774
 dx_abs_0_3_abs_le_orig: dx_abs_0_3 - dx_diff_0_3
   + 774 dx_abs_0_3_is_positive <= 774
 dx_abs_0_3_abs_ge_neg_orig: dx_abs_0_3 + dx_diff_0_3
   + 774 dx_abs_0_3_is_positive >= 0
 dx_abs_0_3_abs_le_neg_orig: dx_abs_0_3 + dx_diff_0_3
   - 774 dx_abs_0_3_is_positive <= 0
 dx_abs_0_3_force_positive: dx_diff_0_3 - 774 dx_abs_0_3_is_positive
   >= -774
 dx_abs_0_3_force_negative: dx_diff_0_3 - 774 dx_abs_0_3_is_positive <= 0
 dy_abs_0_3_abs_ge_orig: dy_abs_0_3 - dy_diff_0_3
   - 774 dy_abs_0_3_is_positive >= -774
 dy_abs_0_3_abs_le_orig: dy_abs_0_3 - dy_diff_0_3
   + 774 dy_abs_0_3_is_positive <= 774
 dy_abs_0_3_abs_ge_neg_orig: dy_abs_0_3 + dy_diff_0_3
   + 774 dy_abs_0_3_is_positive >= 0
 dy_abs_0_3_abs_le_neg_orig: dy_abs_0_3 + dy_diff_0_3
   - 774 dy_abs_0_3_is_positive <= 0
 dy_abs_0_3_force_positive: dy_diff_0_3 - 774 dy_abs_0_3_is_positive
   >= -774
 dy_abs_0_3_force_negative: dy_diff_0_3 - 774 dy_abs_0_3_is_positive <= 0
 dx_diff_def_0_4: - cx_grid_var_0 + cx_grid_var_4 + dx_diff_0_4 = 0
 dy_diff_def_0_4: - cy_grid_var_0 + cy_grid_var_4 + dy_diff_0_4 = 0
 dx_abs_0_4_abs_ge_orig: dx_abs_0_4 - dx_diff_0_4
   - 774 dx_abs_0_4_is_positive >= -774
 dx_abs_0_4_abs_le_orig: dx_abs_0_4 - dx_diff_0_4
   + 774 dx_abs_0_4_is_positive <= 774
 dx_abs_0_4_abs_ge_neg_orig: dx_abs_0_4 + dx_diff_0_4
   + 774 dx_abs_0_4_is_positive >= 0
 dx_abs_0_4_abs_le_neg_orig: dx_abs_0_4 + dx_diff_0_4
   - 774 dx_abs_0_4_is_positive <= 0
 dx_abs_0_4_force_positive: dx_diff_0_4 - 774 dx_abs_0_4_is_positive
   >= -774
 dx_abs_0_4_force_negative: dx_diff_0_4 - 774 dx_abs_0_4_is_positive <= 0
 dy_abs_0_4_abs_ge_orig: dy_abs_0_4 - dy_diff_0_4
   - 774 dy_abs_0_4_is_positive >= -774
 dy_abs_0_4_abs_le_orig: dy_abs_0_4 - dy_diff_0_4
   + 774 dy_abs_0_4_is_positive <= 774
 dy_abs_0_4_abs_ge_neg_orig: dy_abs_0_4 + dy_diff_0_4
   + 774 dy_abs_0_4_is_positive >= 0
 dy_abs_0_4_abs_le_neg_orig: dy_abs_0_4 + dy_diff_0_4
   - 774 dy_abs_0_4_is_positive <= 0
 dy_abs_0_4_force_positive: dy_diff_0_4 - 774 dy_abs_0_4_is_positive
   >= -774
 dy_abs_0_4_force_negative: dy_diff_0_4 - 774 dy_abs_0_4_is_positive <= 0
 dx_diff_def_0_6: - cx_grid_var_0 + cx_grid_var_6 + dx_diff_0_6 = 0
 dy_diff_def_0_6: - cy_grid_var_0 + cy_grid_var_6 + dy_diff_0_6 = 0
 dx_abs_0_6_abs_ge_orig: dx_abs_0_6 - dx_diff_0_6
   - 774 dx_abs_0_6_is_positive >= -774
 dx_abs_0_6_abs_le_orig: dx_abs_0_6 - dx_diff_0_6
   + 774 dx_abs_0_6_is_positive <= 774
 dx_abs_0_6_abs_ge_neg_orig: dx_abs_0_6 + dx_diff_0_6
   + 774 dx_abs_0_6_is_positive >= 0
 dx_abs_0_6_abs_le_neg_orig: dx_abs_0_6 + dx_diff_0_6
   - 774 dx_abs_0_6_is_positive <= 0
 dx_abs_0_6_force_positive: dx_diff_0_6 - 774 dx_abs_0_6_is_positive
   >= -774
 dx_abs_0_6_force_negative: dx_diff_0_6 - 774 dx_abs_0_6_is_positive <= 0
 dy_abs_0_6_abs_ge_orig: dy_abs_0_6 - dy_diff_0_6
   - 774 dy_abs_0_6_is_positive >= -774
 dy_abs_0_6_abs_le_orig: dy_abs_0_6 - dy_diff_0_6
   + 774 dy_abs_0_6_is_positive <= 774
 dy_abs_0_6_abs_ge_neg_orig: dy_abs_0_6 + dy_diff_0_6
   + 774 dy_abs_0_6_is_positive >= 0
 dy_abs_0_6_abs_le_neg_orig: dy_abs_0_6 + dy_diff_0_6
   - 774 dy_abs_0_6_is_positive <= 0
 dy_abs_0_6_force_positive: dy_diff_0_6 - 774 dy_abs_0_6_is_positive
   >= -774
 dy_abs_0_6_force_negative: dy_diff_0_6 - 774 dy_abs_0_6_is_positive <= 0
 dx_diff_def_0_12: - cx_grid_var_0 + cx_grid_var_12 + dx_diff_0_12 = 0
 dy_diff_def_0_12: - cy_grid_var_0 + cy_grid_var_12 + dy_diff_0_12 = 0
 dx_abs_0_12_abs_ge_orig: dx_abs_0_12 - dx_diff_0_12
   - 774 dx_abs_0_12_is_positive >= -774
 dx_abs_0_12_abs_le_orig: dx_abs_0_12 - dx_diff_0_12
   + 774 dx_abs_0_12_is_positive <= 774
 dx_abs_0_12_abs_ge_neg_orig: dx_abs_0_12 + dx_diff_0_12
   + 774 dx_abs_0_12_is_positive >= 0
 dx_abs_0_12_abs_le_neg_orig: dx_abs_0_12 + dx_diff_0_12
   - 774 dx_abs_0_12_is_positive <= 0
 dx_abs_0_12_force_positive: dx_diff_0_12 - 774 dx_abs_0_12_is_positive
   >= -774
 dx_abs_0_12_force_negative: dx_diff_0_12 - 774 dx_abs_0_12_is_positive
   <= 0
 dy_abs_0_12_abs_ge_orig: dy_abs_0_12 - dy_diff_0_12
   - 774 dy_abs_0_12_is_positive >= -774
 dy_abs_0_12_abs_le_orig: dy_abs_0_12 - dy_diff_0_12
   + 774 dy_abs_0_12_is_positive <= 774
 dy_abs_0_12_abs_ge_neg_orig: dy_abs_0_12 + dy_diff_0_12
   + 774 dy_abs_0_12_is_positive >= 0
 dy_abs_0_12_abs_le_neg_orig: dy_abs_0_12 + dy_diff_0_12
   - 774 dy_abs_0_12_is_positive <= 0
 dy_abs_0_12_force_positive: dy_diff_0_12 - 774 dy_abs_0_12_is_positive
   >= -774
 dy_abs_0_12_force_negative: dy_diff_0_12 - 774 dy_abs_0_12_is_positive
   <= 0
 dx_diff_def_0_14: - cx_grid_var_0 + cx_grid_var_14 + dx_diff_0_14 = 0
 dy_diff_def_0_14: - cy_grid_var_0 + cy_grid_var_14 + dy_diff_0_14 = 0
 dx_abs_0_14_abs_ge_orig: dx_abs_0_14 - dx_diff_0_14
   - 774 dx_abs_0_14_is_positive >= -774
 dx_abs_0_14_abs_le_orig: dx_abs_0_14 - dx_diff_0_14
   + 774 dx_abs_0_14_is_positive <= 774
 dx_abs_0_14_abs_ge_neg_orig: dx_abs_0_14 + dx_diff_0_14
   + 774 dx_abs_0_14_is_positive >= 0
 dx_abs_0_14_abs_le_neg_orig: dx_abs_0_14 + dx_diff_0_14
   - 774 dx_abs_0_14_is_positive <= 0
 dx_abs_0_14_force_positive: dx_diff_0_14 - 774 dx_abs_0_14_is_positive
   >= -774
 dx_abs_0_14_force_negative: dx_diff_0_14 - 774 dx_abs_0_14_is_positive
   <= 0
 dy_abs_0_14_abs_ge_orig: dy_abs_0_14 - dy_diff_0_14
   - 774 dy_abs_0_14_is_positive >= -774
 dy_abs_0_14_abs_le_orig: dy_abs_0_14 - dy_diff_0_14
   + 774 dy_abs_0_14_is_positive <= 774
 dy_abs_0_14_abs_ge_neg_orig: dy_abs_0_14 + dy_diff_0_14
   + 774 dy_abs_0_14_is_positive >= 0
 dy_abs_0_14_abs_le_neg_orig: dy_abs_0_14 + dy_diff_0_14
   - 774 dy_abs_0_14_is_positive <= 0
 dy_abs_0_14_force_positive: dy_diff_0_14 - 774 dy_abs_0_14_is_positive
   >= -774
 dy_abs_0_14_force_negative: dy_diff_0_14 - 774 dy_abs_0_14_is_positive
   <= 0
 dx_diff_def_0_15: - cx_grid_var_0 + cx_grid_var_15 + dx_diff_0_15 = 0
 dy_diff_def_0_15: - cy_grid_var_0 + cy_grid_var_15 + dy_diff_0_15 = 0
 dx_abs_0_15_abs_ge_orig: dx_abs_0_15 - dx_diff_0_15
   - 774 dx_abs_0_15_is_positive >= -774
 dx_abs_0_15_abs_le_orig: dx_abs_0_15 - dx_diff_0_15
   + 774 dx_abs_0_15_is_positive <= 774
 dx_abs_0_15_abs_ge_neg_orig: dx_abs_0_15 + dx_diff_0_15
   + 774 dx_abs_0_15_is_positive >= 0
 dx_abs_0_15_abs_le_neg_orig: dx_abs_0_15 + dx_diff_0_15
   - 774 dx_abs_0_15_is_positive <= 0
 dx_abs_0_15_force_positive: dx_diff_0_15 - 774 dx_abs_0_15_is_positive
   >= -774
 dx_abs_0_15_force_negative: dx_diff_0_15 - 774 dx_abs_0_15_is_positive
   <= 0
 dy_abs_0_15_abs_ge_orig: dy_abs_0_15 - dy_diff_0_15
   - 774 dy_abs_0_15_is_positive >= -774
 dy_abs_0_15_abs_le_orig: dy_abs_0_15 - dy_diff_0_15
   + 774 dy_abs_0_15_is_positive <= 774
 dy_abs_0_15_abs_ge_neg_orig: dy_abs_0_15 + dy_diff_0_15
   + 774 dy_abs_0_15_is_positive >= 0
 dy_abs_0_15_abs_le_neg_orig: dy_abs_0_15 + dy_diff_0_15
   - 774 dy_abs_0_15_is_positive <= 0
 dy_abs_0_15_force_positive: dy_diff_0_15 - 774 dy_abs_0_15_is_positive
   >= -774
 dy_abs_0_15_force_negative: dy_diff_0_15 - 774 dy_abs_0_15_is_positive
   <= 0
 dx_diff_def_1_2: - cx_grid_var_1 + cx_grid_var_2 + dx_diff_1_2 = 0
 dy_diff_def_1_2: - cy_grid_var_1 + cy_grid_var_2 + dy_diff_1_2 = 0
 dx_abs_1_2_abs_ge_orig: dx_abs_1_2 - dx_diff_1_2
   - 774 dx_abs_1_2_is_positive >= -774
 dx_abs_1_2_abs_le_orig: dx_abs_1_2 - dx_diff_1_2
   + 774 dx_abs_1_2_is_positive <= 774
 dx_abs_1_2_abs_ge_neg_orig: dx_abs_1_2 + dx_diff_1_2
   + 774 dx_abs_1_2_is_positive >= 0
 dx_abs_1_2_abs_le_neg_orig: dx_abs_1_2 + dx_diff_1_2
   - 774 dx_abs_1_2_is_positive <= 0
 dx_abs_1_2_force_positive: dx_diff_1_2 - 774 dx_abs_1_2_is_positive
   >= -774
 dx_abs_1_2_force_negative: dx_diff_1_2 - 774 dx_abs_1_2_is_positive <= 0
 dy_abs_1_2_abs_ge_orig: dy_abs_1_2 - dy_diff_1_2
   - 774 dy_abs_1_2_is_positive >= -774
 dy_abs_1_2_abs_le_orig: dy_abs_1_2 - dy_diff_1_2
   + 774 dy_abs_1_2_is_positive <= 774
 dy_abs_1_2_abs_ge_neg_orig: dy_abs_1_2 + dy_diff_1_2
   + 774 dy_abs_1_2_is_positive >= 0
 dy_abs_1_2_abs_le_neg_orig: dy_abs_1_2 + dy_diff_1_2
   - 774 dy_abs_1_2_is_positive <= 0
 dy_abs_1_2_force_positive: dy_diff_1_2 - 774 dy_abs_1_2_is_positive
   >= -774
 dy_abs_1_2_force_negative: dy_diff_1_2 - 774 dy_abs_1_2_is_positive <= 0
 dx_diff_def_1_3: - cx_grid_var_1 + cx_grid_var_3 + dx_diff_1_3 = 0
 dy_diff_def_1_3: - cy_grid_var_1 + cy_grid_var_3 + dy_diff_1_3 = 0
 dx_abs_1_3_abs_ge_orig: dx_abs_1_3 - dx_diff_1_3
   - 774 dx_abs_1_3_is_positive >= -774
 dx_abs_1_3_abs_le_orig: dx_abs_1_3 - dx_diff_1_3
   + 774 dx_abs_1_3_is_positive <= 774
 dx_abs_1_3_abs_ge_neg_orig: dx_abs_1_3 + dx_diff_1_3
   + 774 dx_abs_1_3_is_positive >= 0
 dx_abs_1_3_abs_le_neg_orig: dx_abs_1_3 + dx_diff_1_3
   - 774 dx_abs_1_3_is_positive <= 0
 dx_abs_1_3_force_positive: dx_diff_1_3 - 774 dx_abs_1_3_is_positive
   >= -774
 dx_abs_1_3_force_negative: dx_diff_1_3 - 774 dx_abs_1_3_is_positive <= 0
 dy_abs_1_3_abs_ge_orig: dy_abs_1_3 - dy_diff_1_3
   - 774 dy_abs_1_3_is_positive >= -774
 dy_abs_1_3_abs_le_orig: dy_abs_1_3 - dy_diff_1_3
   + 774 dy_abs_1_3_is_positive <= 774
 dy_abs_1_3_abs_ge_neg_orig: dy_abs_1_3 + dy_diff_1_3
   + 774 dy_abs_1_3_is_positive >= 0
 dy_abs_1_3_abs_le_neg_orig: dy_abs_1_3 + dy_diff_1_3
   - 774 dy_abs_1_3_is_positive <= 0
 dy_abs_1_3_force_positive: dy_diff_1_3 - 774 dy_abs_1_3_is_positive
   >= -774
 dy_abs_1_3_force_negative: dy_diff_1_3 - 774 dy_abs_1_3_is_positive <= 0
 dx_diff_def_1_4: - cx_grid_var_1 + cx_grid_var_4 + dx_diff_1_4 = 0
 dy_diff_def_1_4: - cy_grid_var_1 + cy_grid_var_4 + dy_diff_1_4 = 0
 dx_abs_1_4_abs_ge_orig: dx_abs_1_4 - dx_diff_1_4
   - 774 dx_abs_1_4_is_positive >= -774
 dx_abs_1_4_abs_le_orig: dx_abs_1_4 - dx_diff_1_4
   + 774 dx_abs_1_4_is_positive <= 774
 dx_abs_1_4_abs_ge_neg_orig: dx_abs_1_4 + dx_diff_1_4
   + 774 dx_abs_1_4_is_positive >= 0
 dx_abs_1_4_abs_le_neg_orig: dx_abs_1_4 + dx_diff_1_4
   - 774 dx_abs_1_4_is_positive <= 0
 dx_abs_1_4_force_positive: dx_diff_1_4 - 774 dx_abs_1_4_is_positive
   >= -774
 dx_abs_1_4_force_negative: dx_diff_1_4 - 774 dx_abs_1_4_is_positive <= 0
 dy_abs_1_4_abs_ge_orig: dy_abs_1_4 - dy_diff_1_4
   - 774 dy_abs_1_4_is_positive >= -774
 dy_abs_1_4_abs_le_orig: dy_abs_1_4 - dy_diff_1_4
   + 774 dy_abs_1_4_is_positive <= 774
 dy_abs_1_4_abs_ge_neg_orig: dy_abs_1_4 + dy_diff_1_4
   + 774 dy_abs_1_4_is_positive >= 0
 dy_abs_1_4_abs_le_neg_orig: dy_abs_1_4 + dy_diff_1_4
   - 774 dy_abs_1_4_is_positive <= 0
 dy_abs_1_4_force_positive: dy_diff_1_4 - 774 dy_abs_1_4_is_positive
   >= -774
 dy_abs_1_4_force_negative: dy_diff_1_4 - 774 dy_abs_1_4_is_positive <= 0
 dx_diff_def_1_9: - cx_grid_var_1 + cx_grid_var_9 + dx_diff_1_9 = 0
 dy_diff_def_1_9: - cy_grid_var_1 + cy_grid_var_9 + dy_diff_1_9 = 0
 dx_abs_1_9_abs_ge_orig: dx_abs_1_9 - dx_diff_1_9
   - 774 dx_abs_1_9_is_positive >= -774
 dx_abs_1_9_abs_le_orig: dx_abs_1_9 - dx_diff_1_9
   + 774 dx_abs_1_9_is_positive <= 774
 dx_abs_1_9_abs_ge_neg_orig: dx_abs_1_9 + dx_diff_1_9
   + 774 dx_abs_1_9_is_positive >= 0
 dx_abs_1_9_abs_le_neg_orig: dx_abs_1_9 + dx_diff_1_9
   - 774 dx_abs_1_9_is_positive <= 0
 dx_abs_1_9_force_positive: dx_diff_1_9 - 774 dx_abs_1_9_is_positive
   >= -774
 dx_abs_1_9_force_negative: dx_diff_1_9 - 774 dx_abs_1_9_is_positive <= 0
 dy_abs_1_9_abs_ge_orig: dy_abs_1_9 - dy_diff_1_9
   - 774 dy_abs_1_9_is_positive >= -774
 dy_abs_1_9_abs_le_orig: dy_abs_1_9 - dy_diff_1_9
   + 774 dy_abs_1_9_is_positive <= 774
 dy_abs_1_9_abs_ge_neg_orig: dy_abs_1_9 + dy_diff_1_9
   + 774 dy_abs_1_9_is_positive >= 0
 dy_abs_1_9_abs_le_neg_orig: dy_abs_1_9 + dy_diff_1_9
   - 774 dy_abs_1_9_is_positive <= 0
 dy_abs_1_9_force_positive: dy_diff_1_9 - 774 dy_abs_1_9_is_positive
   >= -774
 dy_abs_1_9_force_negative: dy_diff_1_9 - 774 dy_abs_1_9_is_positive <= 0
 dx_diff_def_1_12: - cx_grid_var_1 + cx_grid_var_12 + dx_diff_1_12 = 0
 dy_diff_def_1_12: - cy_grid_var_1 + cy_grid_var_12 + dy_diff_1_12 = 0
 dx_abs_1_12_abs_ge_orig: dx_abs_1_12 - dx_diff_1_12
   - 774 dx_abs_1_12_is_positive >= -774
 dx_abs_1_12_abs_le_orig: dx_abs_1_12 - dx_diff_1_12
   + 774 dx_abs_1_12_is_positive <= 774
 dx_abs_1_12_abs_ge_neg_orig: dx_abs_1_12 + dx_diff_1_12
   + 774 dx_abs_1_12_is_positive >= 0
 dx_abs_1_12_abs_le_neg_orig: dx_abs_1_12 + dx_diff_1_12
   - 774 dx_abs_1_12_is_positive <= 0
 dx_abs_1_12_force_positive: dx_diff_1_12 - 774 dx_abs_1_12_is_positive
   >= -774
 dx_abs_1_12_force_negative: dx_diff_1_12 - 774 dx_abs_1_12_is_positive
   <= 0
 dy_abs_1_12_abs_ge_orig: dy_abs_1_12 - dy_diff_1_12
   - 774 dy_abs_1_12_is_positive >= -774
 dy_abs_1_12_abs_le_orig: dy_abs_1_12 - dy_diff_1_12
   + 774 dy_abs_1_12_is_positive <= 774
 dy_abs_1_12_abs_ge_neg_orig: dy_abs_1_12 + dy_diff_1_12
   + 774 dy_abs_1_12_is_positive >= 0
 dy_abs_1_12_abs_le_neg_orig: dy_abs_1_12 + dy_diff_1_12
   - 774 dy_abs_1_12_is_positive <= 0
 dy_abs_1_12_force_positive: dy_diff_1_12 - 774 dy_abs_1_12_is_positive
   >= -774
 dy_abs_1_12_force_negative: dy_diff_1_12 - 774 dy_abs_1_12_is_positive
   <= 0
 dx_diff_def_1_13: - cx_grid_var_1 + cx_grid_var_13 + dx_diff_1_13 = 0
 dy_diff_def_1_13: - cy_grid_var_1 + cy_grid_var_13 + dy_diff_1_13 = 0
 dx_abs_1_13_abs_ge_orig: dx_abs_1_13 - dx_diff_1_13
   - 774 dx_abs_1_13_is_positive >= -774
 dx_abs_1_13_abs_le_orig: dx_abs_1_13 - dx_diff_1_13
   + 774 dx_abs_1_13_is_positive <= 774
 dx_abs_1_13_abs_ge_neg_orig: dx_abs_1_13 + dx_diff_1_13
   + 774 dx_abs_1_13_is_positive >= 0
 dx_abs_1_13_abs_le_neg_orig: dx_abs_1_13 + dx_diff_1_13
   - 774 dx_abs_1_13_is_positive <= 0
 dx_abs_1_13_force_positive: dx_diff_1_13 - 774 dx_abs_1_13_is_positive
   >= -774
 dx_abs_1_13_force_negative: dx_diff_1_13 - 774 dx_abs_1_13_is_positive
   <= 0
 dy_abs_1_13_abs_ge_orig: dy_abs_1_13 - dy_diff_1_13
   - 774 dy_abs_1_13_is_positive >= -774
 dy_abs_1_13_abs_le_orig: dy_abs_1_13 - dy_diff_1_13
   + 774 dy_abs_1_13_is_positive <= 774
 dy_abs_1_13_abs_ge_neg_orig: dy_abs_1_13 + dy_diff_1_13
   + 774 dy_abs_1_13_is_positive >= 0
 dy_abs_1_13_abs_le_neg_orig: dy_abs_1_13 + dy_diff_1_13
   - 774 dy_abs_1_13_is_positive <= 0
 dy_abs_1_13_force_positive: dy_diff_1_13 - 774 dy_abs_1_13_is_positive
   >= -774
 dy_abs_1_13_force_negative: dy_diff_1_13 - 774 dy_abs_1_13_is_positive
   <= 0
 dx_diff_def_1_14: - cx_grid_var_1 + cx_grid_var_14 + dx_diff_1_14 = 0
 dy_diff_def_1_14: - cy_grid_var_1 + cy_grid_var_14 + dy_diff_1_14 = 0
 dx_abs_1_14_abs_ge_orig: dx_abs_1_14 - dx_diff_1_14
   - 774 dx_abs_1_14_is_positive >= -774
 dx_abs_1_14_abs_le_orig: dx_abs_1_14 - dx_diff_1_14
   + 774 dx_abs_1_14_is_positive <= 774
 dx_abs_1_14_abs_ge_neg_orig: dx_abs_1_14 + dx_diff_1_14
   + 774 dx_abs_1_14_is_positive >= 0
 dx_abs_1_14_abs_le_neg_orig: dx_abs_1_14 + dx_diff_1_14
   - 774 dx_abs_1_14_is_positive <= 0
 dx_abs_1_14_force_positive: dx_diff_1_14 - 774 dx_abs_1_14_is_positive
   >= -774
 dx_abs_1_14_force_negative: dx_diff_1_14 - 774 dx_abs_1_14_is_positive
   <= 0
 dy_abs_1_14_abs_ge_orig: dy_abs_1_14 - dy_diff_1_14
   - 774 dy_abs_1_14_is_positive >= -774
 dy_abs_1_14_abs_le_orig: dy_abs_1_14 - dy_diff_1_14
   + 774 dy_abs_1_14_is_positive <= 774
 dy_abs_1_14_abs_ge_neg_orig: dy_abs_1_14 + dy_diff_1_14
   + 774 dy_abs_1_14_is_positive >= 0
 dy_abs_1_14_abs_le_neg_orig: dy_abs_1_14 + dy_diff_1_14
   - 774 dy_abs_1_14_is_positive <= 0
 dy_abs_1_14_force_positive: dy_diff_1_14 - 774 dy_abs_1_14_is_positive
   >= -774
 dy_abs_1_14_force_negative: dy_diff_1_14 - 774 dy_abs_1_14_is_positive
   <= 0
 dx_diff_def_1_15: - cx_grid_var_1 + cx_grid_var_15 + dx_diff_1_15 = 0
 dy_diff_def_1_15: - cy_grid_var_1 + cy_grid_var_15 + dy_diff_1_15 = 0
 dx_abs_1_15_abs_ge_orig: dx_abs_1_15 - dx_diff_1_15
   - 774 dx_abs_1_15_is_positive >= -774
 dx_abs_1_15_abs_le_orig: dx_abs_1_15 - dx_diff_1_15
   + 774 dx_abs_1_15_is_positive <= 774
 dx_abs_1_15_abs_ge_neg_orig: dx_abs_1_15 + dx_diff_1_15
   + 774 dx_abs_1_15_is_positive >= 0
 dx_abs_1_15_abs_le_neg_orig: dx_abs_1_15 + dx_diff_1_15
   - 774 dx_abs_1_15_is_positive <= 0
 dx_abs_1_15_force_positive: dx_diff_1_15 - 774 dx_abs_1_15_is_positive
   >= -774
 dx_abs_1_15_force_negative: dx_diff_1_15 - 774 dx_abs_1_15_is_positive
   <= 0
 dy_abs_1_15_abs_ge_orig: dy_abs_1_15 - dy_diff_1_15
   - 774 dy_abs_1_15_is_positive >= -774
 dy_abs_1_15_abs_le_orig: dy_abs_1_15 - dy_diff_1_15
   + 774 dy_abs_1_15_is_positive <= 774
 dy_abs_1_15_abs_ge_neg_orig: dy_abs_1_15 + dy_diff_1_15
   + 774 dy_abs_1_15_is_positive >= 0
 dy_abs_1_15_abs_le_neg_orig: dy_abs_1_15 + dy_diff_1_15
   - 774 dy_abs_1_15_is_positive <= 0
 dy_abs_1_15_force_positive: dy_diff_1_15 - 774 dy_abs_1_15_is_positive
   >= -774
 dy_abs_1_15_force_negative: dy_diff_1_15 - 774 dy_abs_1_15_is_positive
   <= 0
 dx_diff_def_2_3: - cx_grid_var_2 + cx_grid_var_3 + dx_diff_2_3 = 0
 dy_diff_def_2_3: - cy_grid_var_2 + cy_grid_var_3 + dy_diff_2_3 = 0
 dx_abs_2_3_abs_ge_orig: dx_abs_2_3 - dx_diff_2_3
   - 774 dx_abs_2_3_is_positive >= -774
 dx_abs_2_3_abs_le_orig: dx_abs_2_3 - dx_diff_2_3
   + 774 dx_abs_2_3_is_positive <= 774
 dx_abs_2_3_abs_ge_neg_orig: dx_abs_2_3 + dx_diff_2_3
   + 774 dx_abs_2_3_is_positive >= 0
 dx_abs_2_3_abs_le_neg_orig: dx_abs_2_3 + dx_diff_2_3
   - 774 dx_abs_2_3_is_positive <= 0
 dx_abs_2_3_force_positive: dx_diff_2_3 - 774 dx_abs_2_3_is_positive
   >= -774
 dx_abs_2_3_force_negative: dx_diff_2_3 - 774 dx_abs_2_3_is_positive <= 0
 dy_abs_2_3_abs_ge_orig: dy_abs_2_3 - dy_diff_2_3
   - 774 dy_abs_2_3_is_positive >= -774
 dy_abs_2_3_abs_le_orig: dy_abs_2_3 - dy_diff_2_3
   + 774 dy_abs_2_3_is_positive <= 774
 dy_abs_2_3_abs_ge_neg_orig: dy_abs_2_3 + dy_diff_2_3
   + 774 dy_abs_2_3_is_positive >= 0
 dy_abs_2_3_abs_le_neg_orig: dy_abs_2_3 + dy_diff_2_3
   - 774 dy_abs_2_3_is_positive <= 0
 dy_abs_2_3_force_positive: dy_diff_2_3 - 774 dy_abs_2_3_is_positive
   >= -774
 dy_abs_2_3_force_negative: dy_diff_2_3 - 774 dy_abs_2_3_is_positive <= 0
 dx_diff_def_2_4: - cx_grid_var_2 + cx_grid_var_4 + dx_diff_2_4 = 0
 dy_diff_def_2_4: - cy_grid_var_2 + cy_grid_var_4 + dy_diff_2_4 = 0
 dx_abs_2_4_abs_ge_orig: dx_abs_2_4 - dx_diff_2_4
   - 774 dx_abs_2_4_is_positive >= -774
 dx_abs_2_4_abs_le_orig: dx_abs_2_4 - dx_diff_2_4
   + 774 dx_abs_2_4_is_positive <= 774
 dx_abs_2_4_abs_ge_neg_orig: dx_abs_2_4 + dx_diff_2_4
   + 774 dx_abs_2_4_is_positive >= 0
 dx_abs_2_4_abs_le_neg_orig: dx_abs_2_4 + dx_diff_2_4
   - 774 dx_abs_2_4_is_positive <= 0
 dx_abs_2_4_force_positive: dx_diff_2_4 - 774 dx_abs_2_4_is_positive
   >= -774
 dx_abs_2_4_force_negative: dx_diff_2_4 - 774 dx_abs_2_4_is_positive <= 0
 dy_abs_2_4_abs_ge_orig: dy_abs_2_4 - dy_diff_2_4
   - 774 dy_abs_2_4_is_positive >= -774
 dy_abs_2_4_abs_le_orig: dy_abs_2_4 - dy_diff_2_4
   + 774 dy_abs_2_4_is_positive <= 774
 dy_abs_2_4_abs_ge_neg_orig: dy_abs_2_4 + dy_diff_2_4
   + 774 dy_abs_2_4_is_positive >= 0
 dy_abs_2_4_abs_le_neg_orig: dy_abs_2_4 + dy_diff_2_4
   - 774 dy_abs_2_4_is_positive <= 0
 dy_abs_2_4_force_positive: dy_diff_2_4 - 774 dy_abs_2_4_is_positive
   >= -774
 dy_abs_2_4_force_negative: dy_diff_2_4 - 774 dy_abs_2_4_is_positive <= 0
 dx_diff_def_2_6: - cx_grid_var_2 + cx_grid_var_6 + dx_diff_2_6 = 0
 dy_diff_def_2_6: - cy_grid_var_2 + cy_grid_var_6 + dy_diff_2_6 = 0
 dx_abs_2_6_abs_ge_orig: dx_abs_2_6 - dx_diff_2_6
   - 774 dx_abs_2_6_is_positive >= -774
 dx_abs_2_6_abs_le_orig: dx_abs_2_6 - dx_diff_2_6
   + 774 dx_abs_2_6_is_positive <= 774
 dx_abs_2_6_abs_ge_neg_orig: dx_abs_2_6 + dx_diff_2_6
   + 774 dx_abs_2_6_is_positive >= 0
 dx_abs_2_6_abs_le_neg_orig: dx_abs_2_6 + dx_diff_2_6
   - 774 dx_abs_2_6_is_positive <= 0
 dx_abs_2_6_force_positive: dx_diff_2_6 - 774 dx_abs_2_6_is_positive
   >= -774
 dx_abs_2_6_force_negative: dx_diff_2_6 - 774 dx_abs_2_6_is_positive <= 0
 dy_abs_2_6_abs_ge_orig: dy_abs_2_6 - dy_diff_2_6
   - 774 dy_abs_2_6_is_positive >= -774
 dy_abs_2_6_abs_le_orig: dy_abs_2_6 - dy_diff_2_6
   + 774 dy_abs_2_6_is_positive <= 774
 dy_abs_2_6_abs_ge_neg_orig: dy_abs_2_6 + dy_diff_2_6
   + 774 dy_abs_2_6_is_positive >= 0
 dy_abs_2_6_abs_le_neg_orig: dy_abs_2_6 + dy_diff_2_6
   - 774 dy_abs_2_6_is_positive <= 0
 dy_abs_2_6_force_positive: dy_diff_2_6 - 774 dy_abs_2_6_is_positive
   >= -774
 dy_abs_2_6_force_negative: dy_diff_2_6 - 774 dy_abs_2_6_is_positive <= 0
 dx_diff_def_2_7: - cx_grid_var_2 + cx_grid_var_7 + dx_diff_2_7 = 0
 dy_diff_def_2_7: - cy_grid_var_2 + cy_grid_var_7 + dy_diff_2_7 = 0
 dx_abs_2_7_abs_ge_orig: dx_abs_2_7 - dx_diff_2_7
   - 774 dx_abs_2_7_is_positive >= -774
 dx_abs_2_7_abs_le_orig: dx_abs_2_7 - dx_diff_2_7
   + 774 dx_abs_2_7_is_positive <= 774
 dx_abs_2_7_abs_ge_neg_orig: dx_abs_2_7 + dx_diff_2_7
   + 774 dx_abs_2_7_is_positive >= 0
 dx_abs_2_7_abs_le_neg_orig: dx_abs_2_7 + dx_diff_2_7
   - 774 dx_abs_2_7_is_positive <= 0
 dx_abs_2_7_force_positive: dx_diff_2_7 - 774 dx_abs_2_7_is_positive
   >= -774
 dx_abs_2_7_force_negative: dx_diff_2_7 - 774 dx_abs_2_7_is_positive <= 0
 dy_abs_2_7_abs_ge_orig: dy_abs_2_7 - dy_diff_2_7
   - 774 dy_abs_2_7_is_positive >= -774
 dy_abs_2_7_abs_le_orig: dy_abs_2_7 - dy_diff_2_7
   + 774 dy_abs_2_7_is_positive <= 774
 dy_abs_2_7_abs_ge_neg_orig: dy_abs_2_7 + dy_diff_2_7
   + 774 dy_abs_2_7_is_positive >= 0
 dy_abs_2_7_abs_le_neg_orig: dy_abs_2_7 + dy_diff_2_7
   - 774 dy_abs_2_7_is_positive <= 0
 dy_abs_2_7_force_positive: dy_diff_2_7 - 774 dy_abs_2_7_is_positive
   >= -774
 dy_abs_2_7_force_negative: dy_diff_2_7 - 774 dy_abs_2_7_is_positive <= 0
 dx_diff_def_2_8: - cx_grid_var_2 + cx_grid_var_8 + dx_diff_2_8 = 0
 dy_diff_def_2_8: - cy_grid_var_2 + cy_grid_var_8 + dy_diff_2_8 = 0
 dx_abs_2_8_abs_ge_orig: dx_abs_2_8 - dx_diff_2_8
   - 774 dx_abs_2_8_is_positive >= -774
 dx_abs_2_8_abs_le_orig: dx_abs_2_8 - dx_diff_2_8
   + 774 dx_abs_2_8_is_positive <= 774
 dx_abs_2_8_abs_ge_neg_orig: dx_abs_2_8 + dx_diff_2_8
   + 774 dx_abs_2_8_is_positive >= 0
 dx_abs_2_8_abs_le_neg_orig: dx_abs_2_8 + dx_diff_2_8
   - 774 dx_abs_2_8_is_positive <= 0
 dx_abs_2_8_force_positive: dx_diff_2_8 - 774 dx_abs_2_8_is_positive
   >= -774
 dx_abs_2_8_force_negative: dx_diff_2_8 - 774 dx_abs_2_8_is_positive <= 0
 dy_abs_2_8_abs_ge_orig: dy_abs_2_8 - dy_diff_2_8
   - 774 dy_abs_2_8_is_positive >= -774
 dy_abs_2_8_abs_le_orig: dy_abs_2_8 - dy_diff_2_8
   + 774 dy_abs_2_8_is_positive <= 774
 dy_abs_2_8_abs_ge_neg_orig: dy_abs_2_8 + dy_diff_2_8
   + 774 dy_abs_2_8_is_positive >= 0
 dy_abs_2_8_abs_le_neg_orig: dy_abs_2_8 + dy_diff_2_8
   - 774 dy_abs_2_8_is_positive <= 0
 dy_abs_2_8_force_positive: dy_diff_2_8 - 774 dy_abs_2_8_is_positive
   >= -774
 dy_abs_2_8_force_negative: dy_diff_2_8 - 774 dy_abs_2_8_is_positive <= 0
 dx_diff_def_2_9: - cx_grid_var_2 + cx_grid_var_9 + dx_diff_2_9 = 0
 dy_diff_def_2_9: - cy_grid_var_2 + cy_grid_var_9 + dy_diff_2_9 = 0
 dx_abs_2_9_abs_ge_orig: dx_abs_2_9 - dx_diff_2_9
   - 774 dx_abs_2_9_is_positive >= -774
 dx_abs_2_9_abs_le_orig: dx_abs_2_9 - dx_diff_2_9
   + 774 dx_abs_2_9_is_positive <= 774
 dx_abs_2_9_abs_ge_neg_orig: dx_abs_2_9 + dx_diff_2_9
   + 774 dx_abs_2_9_is_positive >= 0
 dx_abs_2_9_abs_le_neg_orig: dx_abs_2_9 + dx_diff_2_9
   - 774 dx_abs_2_9_is_positive <= 0
 dx_abs_2_9_force_positive: dx_diff_2_9 - 774 dx_abs_2_9_is_positive
   >= -774
 dx_abs_2_9_force_negative: dx_diff_2_9 - 774 dx_abs_2_9_is_positive <= 0
 dy_abs_2_9_abs_ge_orig: dy_abs_2_9 - dy_diff_2_9
   - 774 dy_abs_2_9_is_positive >= -774
 dy_abs_2_9_abs_le_orig: dy_abs_2_9 - dy_diff_2_9
   + 774 dy_abs_2_9_is_positive <= 774
 dy_abs_2_9_abs_ge_neg_orig: dy_abs_2_9 + dy_diff_2_9
   + 774 dy_abs_2_9_is_positive >= 0
 dy_abs_2_9_abs_le_neg_orig: dy_abs_2_9 + dy_diff_2_9
   - 774 dy_abs_2_9_is_positive <= 0
 dy_abs_2_9_force_positive: dy_diff_2_9 - 774 dy_abs_2_9_is_positive
   >= -774
 dy_abs_2_9_force_negative: dy_diff_2_9 - 774 dy_abs_2_9_is_positive <= 0
 dx_diff_def_2_14: - cx_grid_var_2 + cx_grid_var_14 + dx_diff_2_14 = 0
 dy_diff_def_2_14: - cy_grid_var_2 + cy_grid_var_14 + dy_diff_2_14 = 0
 dx_abs_2_14_abs_ge_orig: dx_abs_2_14 - dx_diff_2_14
   - 774 dx_abs_2_14_is_positive >= -774
 dx_abs_2_14_abs_le_orig: dx_abs_2_14 - dx_diff_2_14
   + 774 dx_abs_2_14_is_positive <= 774
 dx_abs_2_14_abs_ge_neg_orig: dx_abs_2_14 + dx_diff_2_14
   + 774 dx_abs_2_14_is_positive >= 0
 dx_abs_2_14_abs_le_neg_orig: dx_abs_2_14 + dx_diff_2_14
   - 774 dx_abs_2_14_is_positive <= 0
 dx_abs_2_14_force_positive: dx_diff_2_14 - 774 dx_abs_2_14_is_positive
   >= -774
 dx_abs_2_14_force_negative: dx_diff_2_14 - 774 dx_abs_2_14_is_positive
   <= 0
 dy_abs_2_14_abs_ge_orig: dy_abs_2_14 - dy_diff_2_14
   - 774 dy_abs_2_14_is_positive >= -774
 dy_abs_2_14_abs_le_orig: dy_abs_2_14 - dy_diff_2_14
   + 774 dy_abs_2_14_is_positive <= 774
 dy_abs_2_14_abs_ge_neg_orig: dy_abs_2_14 + dy_diff_2_14
   + 774 dy_abs_2_14_is_positive >= 0
 dy_abs_2_14_abs_le_neg_orig: dy_abs_2_14 + dy_diff_2_14
   - 774 dy_abs_2_14_is_positive <= 0
 dy_abs_2_14_force_positive: dy_diff_2_14 - 774 dy_abs_2_14_is_positive
   >= -774
 dy_abs_2_14_force_negative: dy_diff_2_14 - 774 dy_abs_2_14_is_positive
   <= 0
 dx_diff_def_3_4: - cx_grid_var_3 + cx_grid_var_4 + dx_diff_3_4 = 0
 dy_diff_def_3_4: - cy_grid_var_3 + cy_grid_var_4 + dy_diff_3_4 = 0
 dx_abs_3_4_abs_ge_orig: dx_abs_3_4 - dx_diff_3_4
   - 774 dx_abs_3_4_is_positive >= -774
 dx_abs_3_4_abs_le_orig: dx_abs_3_4 - dx_diff_3_4
   + 774 dx_abs_3_4_is_positive <= 774
 dx_abs_3_4_abs_ge_neg_orig: dx_abs_3_4 + dx_diff_3_4
   + 774 dx_abs_3_4_is_positive >= 0
 dx_abs_3_4_abs_le_neg_orig: dx_abs_3_4 + dx_diff_3_4
   - 774 dx_abs_3_4_is_positive <= 0
 dx_abs_3_4_force_positive: dx_diff_3_4 - 774 dx_abs_3_4_is_positive
   >= -774
 dx_abs_3_4_force_negative: dx_diff_3_4 - 774 dx_abs_3_4_is_positive <= 0
 dy_abs_3_4_abs_ge_orig: dy_abs_3_4 - dy_diff_3_4
   - 774 dy_abs_3_4_is_positive >= -774
 dy_abs_3_4_abs_le_orig: dy_abs_3_4 - dy_diff_3_4
   + 774 dy_abs_3_4_is_positive <= 774
 dy_abs_3_4_abs_ge_neg_orig: dy_abs_3_4 + dy_diff_3_4
   + 774 dy_abs_3_4_is_positive >= 0
 dy_abs_3_4_abs_le_neg_orig: dy_abs_3_4 + dy_diff_3_4
   - 774 dy_abs_3_4_is_positive <= 0
 dy_abs_3_4_force_positive: dy_diff_3_4 - 774 dy_abs_3_4_is_positive
   >= -774
 dy_abs_3_4_force_negative: dy_diff_3_4 - 774 dy_abs_3_4_is_positive <= 0
 dx_diff_def_3_6: - cx_grid_var_3 + cx_grid_var_6 + dx_diff_3_6 = 0
 dy_diff_def_3_6: - cy_grid_var_3 + cy_grid_var_6 + dy_diff_3_6 = 0
 dx_abs_3_6_abs_ge_orig: dx_abs_3_6 - dx_diff_3_6
   - 774 dx_abs_3_6_is_positive >= -774
 dx_abs_3_6_abs_le_orig: dx_abs_3_6 - dx_diff_3_6
   + 774 dx_abs_3_6_is_positive <= 774
 dx_abs_3_6_abs_ge_neg_orig: dx_abs_3_6 + dx_diff_3_6
   + 774 dx_abs_3_6_is_positive >= 0
 dx_abs_3_6_abs_le_neg_orig: dx_abs_3_6 + dx_diff_3_6
   - 774 dx_abs_3_6_is_positive <= 0
 dx_abs_3_6_force_positive: dx_diff_3_6 - 774 dx_abs_3_6_is_positive
   >= -774
 dx_abs_3_6_force_negative: dx_diff_3_6 - 774 dx_abs_3_6_is_positive <= 0
 dy_abs_3_6_abs_ge_orig: dy_abs_3_6 - dy_diff_3_6
   - 774 dy_abs_3_6_is_positive >= -774
 dy_abs_3_6_abs_le_orig: dy_abs_3_6 - dy_diff_3_6
   + 774 dy_abs_3_6_is_positive <= 774
 dy_abs_3_6_abs_ge_neg_orig: dy_abs_3_6 + dy_diff_3_6
   + 774 dy_abs_3_6_is_positive >= 0
 dy_abs_3_6_abs_le_neg_orig: dy_abs_3_6 + dy_diff_3_6
   - 774 dy_abs_3_6_is_positive <= 0
 dy_abs_3_6_force_positive: dy_diff_3_6 - 774 dy_abs_3_6_is_positive
   >= -774
 dy_abs_3_6_force_negative: dy_diff_3_6 - 774 dy_abs_3_6_is_positive <= 0
 dx_diff_def_3_11: - cx_grid_var_3 + cx_grid_var_11 + dx_diff_3_11 = 0
 dy_diff_def_3_11: - cy_grid_var_3 + cy_grid_var_11 + dy_diff_3_11 = 0
 dx_abs_3_11_abs_ge_orig: dx_abs_3_11 - dx_diff_3_11
   - 774 dx_abs_3_11_is_positive >= -774
 dx_abs_3_11_abs_le_orig: dx_abs_3_11 - dx_diff_3_11
   + 774 dx_abs_3_11_is_positive <= 774
 dx_abs_3_11_abs_ge_neg_orig: dx_abs_3_11 + dx_diff_3_11
   + 774 dx_abs_3_11_is_positive >= 0
 dx_abs_3_11_abs_le_neg_orig: dx_abs_3_11 + dx_diff_3_11
   - 774 dx_abs_3_11_is_positive <= 0
 dx_abs_3_11_force_positive: dx_diff_3_11 - 774 dx_abs_3_11_is_positive
   >= -774
 dx_abs_3_11_force_negative: dx_diff_3_11 - 774 dx_abs_3_11_is_positive
   <= 0
 dy_abs_3_11_abs_ge_orig: dy_abs_3_11 - dy_diff_3_11
   - 774 dy_abs_3_11_is_positive >= -774
 dy_abs_3_11_abs_le_orig: dy_abs_3_11 - dy_diff_3_11
   + 774 dy_abs_3_11_is_positive <= 774
 dy_abs_3_11_abs_ge_neg_orig: dy_abs_3_11 + dy_diff_3_11
   + 774 dy_abs_3_11_is_positive >= 0
 dy_abs_3_11_abs_le_neg_orig: dy_abs_3_11 + dy_diff_3_11
   - 774 dy_abs_3_11_is_positive <= 0
 dy_abs_3_11_force_positive: dy_diff_3_11 - 774 dy_abs_3_11_is_positive
   >= -774
 dy_abs_3_11_force_negative: dy_diff_3_11 - 774 dy_abs_3_11_is_positive
   <= 0
 dx_diff_def_3_13: - cx_grid_var_3 + cx_grid_var_13 + dx_diff_3_13 = 0
 dy_diff_def_3_13: - cy_grid_var_3 + cy_grid_var_13 + dy_diff_3_13 = 0
 dx_abs_3_13_abs_ge_orig: dx_abs_3_13 - dx_diff_3_13
   - 774 dx_abs_3_13_is_positive >= -774
 dx_abs_3_13_abs_le_orig: dx_abs_3_13 - dx_diff_3_13
   + 774 dx_abs_3_13_is_positive <= 774
 dx_abs_3_13_abs_ge_neg_orig: dx_abs_3_13 + dx_diff_3_13
   + 774 dx_abs_3_13_is_positive >= 0
 dx_abs_3_13_abs_le_neg_orig: dx_abs_3_13 + dx_diff_3_13
   - 774 dx_abs_3_13_is_positive <= 0
 dx_abs_3_13_force_positive: dx_diff_3_13 - 774 dx_abs_3_13_is_positive
   >= -774
 dx_abs_3_13_force_negative: dx_diff_3_13 - 774 dx_abs_3_13_is_positive
   <= 0
 dy_abs_3_13_abs_ge_orig: dy_abs_3_13 - dy_diff_3_13
   - 774 dy_abs_3_13_is_positive >= -774
 dy_abs_3_13_abs_le_orig: dy_abs_3_13 - dy_diff_3_13
   + 774 dy_abs_3_13_is_positive <= 774
 dy_abs_3_13_abs_ge_neg_orig: dy_abs_3_13 + dy_diff_3_13
   + 774 dy_abs_3_13_is_positive >= 0
 dy_abs_3_13_abs_le_neg_orig: dy_abs_3_13 + dy_diff_3_13
   - 774 dy_abs_3_13_is_positive <= 0
 dy_abs_3_13_force_positive: dy_diff_3_13 - 774 dy_abs_3_13_is_positive
   >= -774
 dy_abs_3_13_force_negative: dy_diff_3_13 - 774 dy_abs_3_13_is_positive
   <= 0
 dx_diff_def_3_14: - cx_grid_var_3 + cx_grid_var_14 + dx_diff_3_14 = 0
 dy_diff_def_3_14: - cy_grid_var_3 + cy_grid_var_14 + dy_diff_3_14 = 0
 dx_abs_3_14_abs_ge_orig: dx_abs_3_14 - dx_diff_3_14
   - 774 dx_abs_3_14_is_positive >= -774
 dx_abs_3_14_abs_le_orig: dx_abs_3_14 - dx_diff_3_14
   + 774 dx_abs_3_14_is_positive <= 774
 dx_abs_3_14_abs_ge_neg_orig: dx_abs_3_14 + dx_diff_3_14
   + 774 dx_abs_3_14_is_positive >= 0
 dx_abs_3_14_abs_le_neg_orig: dx_abs_3_14 + dx_diff_3_14
   - 774 dx_abs_3_14_is_positive <= 0
 dx_abs_3_14_force_positive: dx_diff_3_14 - 774 dx_abs_3_14_is_positive
   >= -774
 dx_abs_3_14_force_negative: dx_diff_3_14 - 774 dx_abs_3_14_is_positive
   <= 0
 dy_abs_3_14_abs_ge_orig: dy_abs_3_14 - dy_diff_3_14
   - 774 dy_abs_3_14_is_positive >= -774
 dy_abs_3_14_abs_le_orig: dy_abs_3_14 - dy_diff_3_14
   + 774 dy_abs_3_14_is_positive <= 774
 dy_abs_3_14_abs_ge_neg_orig: dy_abs_3_14 + dy_diff_3_14
   + 774 dy_abs_3_14_is_positive >= 0
 dy_abs_3_14_abs_le_neg_orig: dy_abs_3_14 + dy_diff_3_14
   - 774 dy_abs_3_14_is_positive <= 0
 dy_abs_3_14_force_positive: dy_diff_3_14 - 774 dy_abs_3_14_is_positive
   >= -774
 dy_abs_3_14_force_negative: dy_diff_3_14 - 774 dy_abs_3_14_is_positive
   <= 0
 dx_diff_def_4_6: - cx_grid_var_4 + cx_grid_var_6 + dx_diff_4_6 = 0
 dy_diff_def_4_6: - cy_grid_var_4 + cy_grid_var_6 + dy_diff_4_6 = 0
 dx_abs_4_6_abs_ge_orig: dx_abs_4_6 - dx_diff_4_6
   - 774 dx_abs_4_6_is_positive >= -774
 dx_abs_4_6_abs_le_orig: dx_abs_4_6 - dx_diff_4_6
   + 774 dx_abs_4_6_is_positive <= 774
 dx_abs_4_6_abs_ge_neg_orig: dx_abs_4_6 + dx_diff_4_6
   + 774 dx_abs_4_6_is_positive >= 0
 dx_abs_4_6_abs_le_neg_orig: dx_abs_4_6 + dx_diff_4_6
   - 774 dx_abs_4_6_is_positive <= 0
 dx_abs_4_6_force_positive: dx_diff_4_6 - 774 dx_abs_4_6_is_positive
   >= -774
 dx_abs_4_6_force_negative: dx_diff_4_6 - 774 dx_abs_4_6_is_positive <= 0
 dy_abs_4_6_abs_ge_orig: dy_abs_4_6 - dy_diff_4_6
   - 774 dy_abs_4_6_is_positive >= -774
 dy_abs_4_6_abs_le_orig: dy_abs_4_6 - dy_diff_4_6
   + 774 dy_abs_4_6_is_positive <= 774
 dy_abs_4_6_abs_ge_neg_orig: dy_abs_4_6 + dy_diff_4_6
   + 774 dy_abs_4_6_is_positive >= 0
 dy_abs_4_6_abs_le_neg_orig: dy_abs_4_6 + dy_diff_4_6
   - 774 dy_abs_4_6_is_positive <= 0
 dy_abs_4_6_force_positive: dy_diff_4_6 - 774 dy_abs_4_6_is_positive
   >= -774
 dy_abs_4_6_force_negative: dy_diff_4_6 - 774 dy_abs_4_6_is_positive <= 0
 dx_diff_def_4_11: - cx_grid_var_4 + cx_grid_var_11 + dx_diff_4_11 = 0
 dy_diff_def_4_11: - cy_grid_var_4 + cy_grid_var_11 + dy_diff_4_11 = 0
 dx_abs_4_11_abs_ge_orig: dx_abs_4_11 - dx_diff_4_11
   - 774 dx_abs_4_11_is_positive >= -774
 dx_abs_4_11_abs_le_orig: dx_abs_4_11 - dx_diff_4_11
   + 774 dx_abs_4_11_is_positive <= 774
 dx_abs_4_11_abs_ge_neg_orig: dx_abs_4_11 + dx_diff_4_11
   + 774 dx_abs_4_11_is_positive >= 0
 dx_abs_4_11_abs_le_neg_orig: dx_abs_4_11 + dx_diff_4_11
   - 774 dx_abs_4_11_is_positive <= 0
 dx_abs_4_11_force_positive: dx_diff_4_11 - 774 dx_abs_4_11_is_positive
   >= -774
 dx_abs_4_11_force_negative: dx_diff_4_11 - 774 dx_abs_4_11_is_positive
   <= 0
 dy_abs_4_11_abs_ge_orig: dy_abs_4_11 - dy_diff_4_11
   - 774 dy_abs_4_11_is_positive >= -774
 dy_abs_4_11_abs_le_orig: dy_abs_4_11 - dy_diff_4_11
   + 774 dy_abs_4_11_is_positive <= 774
 dy_abs_4_11_abs_ge_neg_orig: dy_abs_4_11 + dy_diff_4_11
   + 774 dy_abs_4_11_is_positive >= 0
 dy_abs_4_11_abs_le_neg_orig: dy_abs_4_11 + dy_diff_4_11
   - 774 dy_abs_4_11_is_positive <= 0
 dy_abs_4_11_force_positive: dy_diff_4_11 - 774 dy_abs_4_11_is_positive
   >= -774
 dy_abs_4_11_force_negative: dy_diff_4_11 - 774 dy_abs_4_11_is_positive
   <= 0
 dx_diff_def_4_15: - cx_grid_var_4 + cx_grid_var_15 + dx_diff_4_15 = 0
 dy_diff_def_4_15: - cy_grid_var_4 + cy_grid_var_15 + dy_diff_4_15 = 0
 dx_abs_4_15_abs_ge_orig: dx_abs_4_15 - dx_diff_4_15
   - 774 dx_abs_4_15_is_positive >= -774
 dx_abs_4_15_abs_le_orig: dx_abs_4_15 - dx_diff_4_15
   + 774 dx_abs_4_15_is_positive <= 774
 dx_abs_4_15_abs_ge_neg_orig: dx_abs_4_15 + dx_diff_4_15
   + 774 dx_abs_4_15_is_positive >= 0
 dx_abs_4_15_abs_le_neg_orig: dx_abs_4_15 + dx_diff_4_15
   - 774 dx_abs_4_15_is_positive <= 0
 dx_abs_4_15_force_positive: dx_diff_4_15 - 774 dx_abs_4_15_is_positive
   >= -774
 dx_abs_4_15_force_negative: dx_diff_4_15 - 774 dx_abs_4_15_is_positive
   <= 0
 dy_abs_4_15_abs_ge_orig: dy_abs_4_15 - dy_diff_4_15
   - 774 dy_abs_4_15_is_positive >= -774
 dy_abs_4_15_abs_le_orig: dy_abs_4_15 - dy_diff_4_15
   + 774 dy_abs_4_15_is_positive <= 774
 dy_abs_4_15_abs_ge_neg_orig: dy_abs_4_15 + dy_diff_4_15
   + 774 dy_abs_4_15_is_positive >= 0
 dy_abs_4_15_abs_le_neg_orig: dy_abs_4_15 + dy_diff_4_15
   - 774 dy_abs_4_15_is_positive <= 0
 dy_abs_4_15_force_positive: dy_diff_4_15 - 774 dy_abs_4_15_is_positive
   >= -774
 dy_abs_4_15_force_negative: dy_diff_4_15 - 774 dy_abs_4_15_is_positive
   <= 0
 dx_diff_def_5_9: - cx_grid_var_5 + cx_grid_var_9 + dx_diff_5_9 = 0
 dy_diff_def_5_9: - cy_grid_var_5 + cy_grid_var_9 + dy_diff_5_9 = 0
 dx_abs_5_9_abs_ge_orig: dx_abs_5_9 - dx_diff_5_9
   - 774 dx_abs_5_9_is_positive >= -774
 dx_abs_5_9_abs_le_orig: dx_abs_5_9 - dx_diff_5_9
   + 774 dx_abs_5_9_is_positive <= 774
 dx_abs_5_9_abs_ge_neg_orig: dx_abs_5_9 + dx_diff_5_9
   + 774 dx_abs_5_9_is_positive >= 0
 dx_abs_5_9_abs_le_neg_orig: dx_abs_5_9 + dx_diff_5_9
   - 774 dx_abs_5_9_is_positive <= 0
 dx_abs_5_9_force_positive: dx_diff_5_9 - 774 dx_abs_5_9_is_positive
   >= -774
 dx_abs_5_9_force_negative: dx_diff_5_9 - 774 dx_abs_5_9_is_positive <= 0
 dy_abs_5_9_abs_ge_orig: dy_abs_5_9 - dy_diff_5_9
   - 774 dy_abs_5_9_is_positive >= -774
 dy_abs_5_9_abs_le_orig: dy_abs_5_9 - dy_diff_5_9
   + 774 dy_abs_5_9_is_positive <= 774
 dy_abs_5_9_abs_ge_neg_orig: dy_abs_5_9 + dy_diff_5_9
   + 774 dy_abs_5_9_is_positive >= 0
 dy_abs_5_9_abs_le_neg_orig: dy_abs_5_9 + dy_diff_5_9
   - 774 dy_abs_5_9_is_positive <= 0
 dy_abs_5_9_force_positive: dy_diff_5_9 - 774 dy_abs_5_9_is_positive
   >= -774
 dy_abs_5_9_force_negative: dy_diff_5_9 - 774 dy_abs_5_9_is_positive <= 0
 dx_diff_def_5_10: - cx_grid_var_5 + cx_grid_var_10 + dx_diff_5_10 = 0
 dy_diff_def_5_10: - cy_grid_var_5 + cy_grid_var_10 + dy_diff_5_10 = 0
 dx_abs_5_10_abs_ge_orig: dx_abs_5_10 - dx_diff_5_10
   - 774 dx_abs_5_10_is_positive >= -774
 dx_abs_5_10_abs_le_orig: dx_abs_5_10 - dx_diff_5_10
   + 774 dx_abs_5_10_is_positive <= 774
 dx_abs_5_10_abs_ge_neg_orig: dx_abs_5_10 + dx_diff_5_10
   + 774 dx_abs_5_10_is_positive >= 0
 dx_abs_5_10_abs_le_neg_orig: dx_abs_5_10 + dx_diff_5_10
   - 774 dx_abs_5_10_is_positive <= 0
 dx_abs_5_10_force_positive: dx_diff_5_10 - 774 dx_abs_5_10_is_positive
   >= -774
 dx_abs_5_10_force_negative: dx_diff_5_10 - 774 dx_abs_5_10_is_positive
   <= 0
 dy_abs_5_10_abs_ge_orig: dy_abs_5_10 - dy_diff_5_10
   - 774 dy_abs_5_10_is_positive >= -774
 dy_abs_5_10_abs_le_orig: dy_abs_5_10 - dy_diff_5_10
   + 774 dy_abs_5_10_is_positive <= 774
 dy_abs_5_10_abs_ge_neg_orig: dy_abs_5_10 + dy_diff_5_10
   + 774 dy_abs_5_10_is_positive >= 0
 dy_abs_5_10_abs_le_neg_orig: dy_abs_5_10 + dy_diff_5_10
   - 774 dy_abs_5_10_is_positive <= 0
 dy_abs_5_10_force_positive: dy_diff_5_10 - 774 dy_abs_5_10_is_positive
   >= -774
 dy_abs_5_10_force_negative: dy_diff_5_10 - 774 dy_abs_5_10_is_positive
   <= 0
 dx_diff_def_6_10: - cx_grid_var_6 + cx_grid_var_10 + dx_diff_6_10 = 0
 dy_diff_def_6_10: - cy_grid_var_6 + cy_grid_var_10 + dy_diff_6_10 = 0
 dx_abs_6_10_abs_ge_orig: dx_abs_6_10 - dx_diff_6_10
   - 774 dx_abs_6_10_is_positive >= -774
 dx_abs_6_10_abs_le_orig: dx_abs_6_10 - dx_diff_6_10
   + 774 dx_abs_6_10_is_positive <= 774
 dx_abs_6_10_abs_ge_neg_orig: dx_abs_6_10 + dx_diff_6_10
   + 774 dx_abs_6_10_is_positive >= 0
 dx_abs_6_10_abs_le_neg_orig: dx_abs_6_10 + dx_diff_6_10
   - 774 dx_abs_6_10_is_positive <= 0
 dx_abs_6_10_force_positive: dx_diff_6_10 - 774 dx_abs_6_10_is_positive
   >= -774
 dx_abs_6_10_force_negative: dx_diff_6_10 - 774 dx_abs_6_10_is_positive
   <= 0
 dy_abs_6_10_abs_ge_orig: dy_abs_6_10 - dy_diff_6_10
   - 774 dy_abs_6_10_is_positive >= -774
 dy_abs_6_10_abs_le_orig: dy_abs_6_10 - dy_diff_6_10
   + 774 dy_abs_6_10_is_positive <= 774
 dy_abs_6_10_abs_ge_neg_orig: dy_abs_6_10 + dy_diff_6_10
   + 774 dy_abs_6_10_is_positive >= 0
 dy_abs_6_10_abs_le_neg_orig: dy_abs_6_10 + dy_diff_6_10
   - 774 dy_abs_6_10_is_positive <= 0
 dy_abs_6_10_force_positive: dy_diff_6_10 - 774 dy_abs_6_10_is_positive
   >= -774
 dy_abs_6_10_force_negative: dy_diff_6_10 - 774 dy_abs_6_10_is_positive
   <= 0
 dx_diff_def_6_12: - cx_grid_var_6 + cx_grid_var_12 + dx_diff_6_12 = 0
 dy_diff_def_6_12: - cy_grid_var_6 + cy_grid_var_12 + dy_diff_6_12 = 0
 dx_abs_6_12_abs_ge_orig: dx_abs_6_12 - dx_diff_6_12
   - 774 dx_abs_6_12_is_positive >= -774
 dx_abs_6_12_abs_le_orig: dx_abs_6_12 - dx_diff_6_12
   + 774 dx_abs_6_12_is_positive <= 774
 dx_abs_6_12_abs_ge_neg_orig: dx_abs_6_12 + dx_diff_6_12
   + 774 dx_abs_6_12_is_positive >= 0
 dx_abs_6_12_abs_le_neg_orig: dx_abs_6_12 + dx_diff_6_12
   - 774 dx_abs_6_12_is_positive <= 0
 dx_abs_6_12_force_positive: dx_diff_6_12 - 774 dx_abs_6_12_is_positive
   >= -774
 dx_abs_6_12_force_negative: dx_diff_6_12 - 774 dx_abs_6_12_is_positive
   <= 0
 dy_abs_6_12_abs_ge_orig: dy_abs_6_12 - dy_diff_6_12
   - 774 dy_abs_6_12_is_positive >= -774
 dy_abs_6_12_abs_le_orig: dy_abs_6_12 - dy_diff_6_12
   + 774 dy_abs_6_12_is_positive <= 774
 dy_abs_6_12_abs_ge_neg_orig: dy_abs_6_12 + dy_diff_6_12
   + 774 dy_abs_6_12_is_positive >= 0
 dy_abs_6_12_abs_le_neg_orig: dy_abs_6_12 + dy_diff_6_12
   - 774 dy_abs_6_12_is_positive <= 0
 dy_abs_6_12_force_positive: dy_diff_6_12 - 774 dy_abs_6_12_is_positive
   >= -774
 dy_abs_6_12_force_negative: dy_diff_6_12 - 774 dy_abs_6_12_is_positive
   <= 0
 dx_diff_def_6_13: - cx_grid_var_6 + cx_grid_var_13 + dx_diff_6_13 = 0
 dy_diff_def_6_13: - cy_grid_var_6 + cy_grid_var_13 + dy_diff_6_13 = 0
 dx_abs_6_13_abs_ge_orig: dx_abs_6_13 - dx_diff_6_13
   - 774 dx_abs_6_13_is_positive >= -774
 dx_abs_6_13_abs_le_orig: dx_abs_6_13 - dx_diff_6_13
   + 774 dx_abs_6_13_is_positive <= 774
 dx_abs_6_13_abs_ge_neg_orig: dx_abs_6_13 + dx_diff_6_13
   + 774 dx_abs_6_13_is_positive >= 0
 dx_abs_6_13_abs_le_neg_orig: dx_abs_6_13 + dx_diff_6_13
   - 774 dx_abs_6_13_is_positive <= 0
 dx_abs_6_13_force_positive: dx_diff_6_13 - 774 dx_abs_6_13_is_positive
   >= -774
 dx_abs_6_13_force_negative: dx_diff_6_13 - 774 dx_abs_6_13_is_positive
   <= 0
 dy_abs_6_13_abs_ge_orig: dy_abs_6_13 - dy_diff_6_13
   - 774 dy_abs_6_13_is_positive >= -774
 dy_abs_6_13_abs_le_orig: dy_abs_6_13 - dy_diff_6_13
   + 774 dy_abs_6_13_is_positive <= 774
 dy_abs_6_13_abs_ge_neg_orig: dy_abs_6_13 + dy_diff_6_13
   + 774 dy_abs_6_13_is_positive >= 0
 dy_abs_6_13_abs_le_neg_orig: dy_abs_6_13 + dy_diff_6_13
   - 774 dy_abs_6_13_is_positive <= 0
 dy_abs_6_13_force_positive: dy_diff_6_13 - 774 dy_abs_6_13_is_positive
   >= -774
 dy_abs_6_13_force_negative: dy_diff_6_13 - 774 dy_abs_6_13_is_positive
   <= 0
 dx_diff_def_6_14: - cx_grid_var_6 + cx_grid_var_14 + dx_diff_6_14 = 0
 dy_diff_def_6_14: - cy_grid_var_6 + cy_grid_var_14 + dy_diff_6_14 = 0
 dx_abs_6_14_abs_ge_orig: dx_abs_6_14 - dx_diff_6_14
   - 774 dx_abs_6_14_is_positive >= -774
 dx_abs_6_14_abs_le_orig: dx_abs_6_14 - dx_diff_6_14
   + 774 dx_abs_6_14_is_positive <= 774
 dx_abs_6_14_abs_ge_neg_orig: dx_abs_6_14 + dx_diff_6_14
   + 774 dx_abs_6_14_is_positive >= 0
 dx_abs_6_14_abs_le_neg_orig: dx_abs_6_14 + dx_diff_6_14
   - 774 dx_abs_6_14_is_positive <= 0
 dx_abs_6_14_force_positive: dx_diff_6_14 - 774 dx_abs_6_14_is_positive
   >= -774
 dx_abs_6_14_force_negative: dx_diff_6_14 - 774 dx_abs_6_14_is_positive
   <= 0
 dy_abs_6_14_abs_ge_orig: dy_abs_6_14 - dy_diff_6_14
   - 774 dy_abs_6_14_is_positive >= -774
 dy_abs_6_14_abs_le_orig: dy_abs_6_14 - dy_diff_6_14
   + 774 dy_abs_6_14_is_positive <= 774
 dy_abs_6_14_abs_ge_neg_orig: dy_abs_6_14 + dy_diff_6_14
   + 774 dy_abs_6_14_is_positive >= 0
 dy_abs_6_14_abs_le_neg_orig: dy_abs_6_14 + dy_diff_6_14
   - 774 dy_abs_6_14_is_positive <= 0
 dy_abs_6_14_force_positive: dy_diff_6_14 - 774 dy_abs_6_14_is_positive
   >= -774
 dy_abs_6_14_force_negative: dy_diff_6_14 - 774 dy_abs_6_14_is_positive
   <= 0
 dx_diff_def_7_11: - cx_grid_var_7 + cx_grid_var_11 + dx_diff_7_11 = 0
 dy_diff_def_7_11: - cy_grid_var_7 + cy_grid_var_11 + dy_diff_7_11 = 0
 dx_abs_7_11_abs_ge_orig: dx_abs_7_11 - dx_diff_7_11
   - 774 dx_abs_7_11_is_positive >= -774
 dx_abs_7_11_abs_le_orig: dx_abs_7_11 - dx_diff_7_11
   + 774 dx_abs_7_11_is_positive <= 774
 dx_abs_7_11_abs_ge_neg_orig: dx_abs_7_11 + dx_diff_7_11
   + 774 dx_abs_7_11_is_positive >= 0
 dx_abs_7_11_abs_le_neg_orig: dx_abs_7_11 + dx_diff_7_11
   - 774 dx_abs_7_11_is_positive <= 0
 dx_abs_7_11_force_positive: dx_diff_7_11 - 774 dx_abs_7_11_is_positive
   >= -774
 dx_abs_7_11_force_negative: dx_diff_7_11 - 774 dx_abs_7_11_is_positive
   <= 0
 dy_abs_7_11_abs_ge_orig: dy_abs_7_11 - dy_diff_7_11
   - 774 dy_abs_7_11_is_positive >= -774
 dy_abs_7_11_abs_le_orig: dy_abs_7_11 - dy_diff_7_11
   + 774 dy_abs_7_11_is_positive <= 774
 dy_abs_7_11_abs_ge_neg_orig: dy_abs_7_11 + dy_diff_7_11
   + 774 dy_abs_7_11_is_positive >= 0
 dy_abs_7_11_abs_le_neg_orig: dy_abs_7_11 + dy_diff_7_11
   - 774 dy_abs_7_11_is_positive <= 0
 dy_abs_7_11_force_positive: dy_diff_7_11 - 774 dy_abs_7_11_is_positive
   >= -774
 dy_abs_7_11_force_negative: dy_diff_7_11 - 774 dy_abs_7_11_is_positive
   <= 0
 dx_diff_def_7_12: - cx_grid_var_7 + cx_grid_var_12 + dx_diff_7_12 = 0
 dy_diff_def_7_12: - cy_grid_var_7 + cy_grid_var_12 + dy_diff_7_12 = 0
 dx_abs_7_12_abs_ge_orig: dx_abs_7_12 - dx_diff_7_12
   - 774 dx_abs_7_12_is_positive >= -774
 dx_abs_7_12_abs_le_orig: dx_abs_7_12 - dx_diff_7_12
   + 774 dx_abs_7_12_is_positive <= 774
 dx_abs_7_12_abs_ge_neg_orig: dx_abs_7_12 + dx_diff_7_12
   + 774 dx_abs_7_12_is_positive >= 0
 dx_abs_7_12_abs_le_neg_orig: dx_abs_7_12 + dx_diff_7_12
   - 774 dx_abs_7_12_is_positive <= 0
 dx_abs_7_12_force_positive: dx_diff_7_12 - 774 dx_abs_7_12_is_positive
   >= -774
 dx_abs_7_12_force_negative: dx_diff_7_12 - 774 dx_abs_7_12_is_positive
   <= 0
 dy_abs_7_12_abs_ge_orig: dy_abs_7_12 - dy_diff_7_12
   - 774 dy_abs_7_12_is_positive >= -774
 dy_abs_7_12_abs_le_orig: dy_abs_7_12 - dy_diff_7_12
   + 774 dy_abs_7_12_is_positive <= 774
 dy_abs_7_12_abs_ge_neg_orig: dy_abs_7_12 + dy_diff_7_12
   + 774 dy_abs_7_12_is_positive >= 0
 dy_abs_7_12_abs_le_neg_orig: dy_abs_7_12 + dy_diff_7_12
   - 774 dy_abs_7_12_is_positive <= 0
 dy_abs_7_12_force_positive: dy_diff_7_12 - 774 dy_abs_7_12_is_positive
   >= -774
 dy_abs_7_12_force_negative: dy_diff_7_12 - 774 dy_abs_7_12_is_positive
   <= 0
 dx_diff_def_7_13: - cx_grid_var_7 + cx_grid_var_13 + dx_diff_7_13 = 0
 dy_diff_def_7_13: - cy_grid_var_7 + cy_grid_var_13 + dy_diff_7_13 = 0
 dx_abs_7_13_abs_ge_orig: dx_abs_7_13 - dx_diff_7_13
   - 774 dx_abs_7_13_is_positive >= -774
 dx_abs_7_13_abs_le_orig: dx_abs_7_13 - dx_diff_7_13
   + 774 dx_abs_7_13_is_positive <= 774
 dx_abs_7_13_abs_ge_neg_orig: dx_abs_7_13 + dx_diff_7_13
   + 774 dx_abs_7_13_is_positive >= 0
 dx_abs_7_13_abs_le_neg_orig: dx_abs_7_13 + dx_diff_7_13
   - 774 dx_abs_7_13_is_positive <= 0
 dx_abs_7_13_force_positive: dx_diff_7_13 - 774 dx_abs_7_13_is_positive
   >= -774
 dx_abs_7_13_force_negative: dx_diff_7_13 - 774 dx_abs_7_13_is_positive
   <= 0
 dy_abs_7_13_abs_ge_orig: dy_abs_7_13 - dy_diff_7_13
   - 774 dy_abs_7_13_is_positive >= -774
 dy_abs_7_13_abs_le_orig: dy_abs_7_13 - dy_diff_7_13
   + 774 dy_abs_7_13_is_positive <= 774
 dy_abs_7_13_abs_ge_neg_orig: dy_abs_7_13 + dy_diff_7_13
   + 774 dy_abs_7_13_is_positive >= 0
 dy_abs_7_13_abs_le_neg_orig: dy_abs_7_13 + dy_diff_7_13
   - 774 dy_abs_7_13_is_positive <= 0
 dy_abs_7_13_force_positive: dy_diff_7_13 - 774 dy_abs_7_13_is_positive
   >= -774
 dy_abs_7_13_force_negative: dy_diff_7_13 - 774 dy_abs_7_13_is_positive
   <= 0
 dx_diff_def_7_14: - cx_grid_var_7 + cx_grid_var_14 + dx_diff_7_14 = 0
 dy_diff_def_7_14: - cy_grid_var_7 + cy_grid_var_14 + dy_diff_7_14 = 0
 dx_abs_7_14_abs_ge_orig: dx_abs_7_14 - dx_diff_7_14
   - 774 dx_abs_7_14_is_positive >= -774
 dx_abs_7_14_abs_le_orig: dx_abs_7_14 - dx_diff_7_14
   + 774 dx_abs_7_14_is_positive <= 774
 dx_abs_7_14_abs_ge_neg_orig: dx_abs_7_14 + dx_diff_7_14
   + 774 dx_abs_7_14_is_positive >= 0
 dx_abs_7_14_abs_le_neg_orig: dx_abs_7_14 + dx_diff_7_14
   - 774 dx_abs_7_14_is_positive <= 0
 dx_abs_7_14_force_positive: dx_diff_7_14 - 774 dx_abs_7_14_is_positive
   >= -774
 dx_abs_7_14_force_negative: dx_diff_7_14 - 774 dx_abs_7_14_is_positive
   <= 0
 dy_abs_7_14_abs_ge_orig: dy_abs_7_14 - dy_diff_7_14
   - 774 dy_abs_7_14_is_positive >= -774
 dy_abs_7_14_abs_le_orig: dy_abs_7_14 - dy_diff_7_14
   + 774 dy_abs_7_14_is_positive <= 774
 dy_abs_7_14_abs_ge_neg_orig: dy_abs_7_14 + dy_diff_7_14
   + 774 dy_abs_7_14_is_positive >= 0
 dy_abs_7_14_abs_le_neg_orig: dy_abs_7_14 + dy_diff_7_14
   - 774 dy_abs_7_14_is_positive <= 0
 dy_abs_7_14_force_positive: dy_diff_7_14 - 774 dy_abs_7_14_is_positive
   >= -774
 dy_abs_7_14_force_negative: dy_diff_7_14 - 774 dy_abs_7_14_is_positive
   <= 0
 dx_diff_def_8_9: - cx_grid_var_8 + cx_grid_var_9 + dx_diff_8_9 = 0
 dy_diff_def_8_9: - cy_grid_var_8 + cy_grid_var_9 + dy_diff_8_9 = 0
 dx_abs_8_9_abs_ge_orig: dx_abs_8_9 - dx_diff_8_9
   - 774 dx_abs_8_9_is_positive >= -774
 dx_abs_8_9_abs_le_orig: dx_abs_8_9 - dx_diff_8_9
   + 774 dx_abs_8_9_is_positive <= 774
 dx_abs_8_9_abs_ge_neg_orig: dx_abs_8_9 + dx_diff_8_9
   + 774 dx_abs_8_9_is_positive >= 0
 dx_abs_8_9_abs_le_neg_orig: dx_abs_8_9 + dx_diff_8_9
   - 774 dx_abs_8_9_is_positive <= 0
 dx_abs_8_9_force_positive: dx_diff_8_9 - 774 dx_abs_8_9_is_positive
   >= -774
 dx_abs_8_9_force_negative: dx_diff_8_9 - 774 dx_abs_8_9_is_positive <= 0
 dy_abs_8_9_abs_ge_orig: dy_abs_8_9 - dy_diff_8_9
   - 774 dy_abs_8_9_is_positive >= -774
 dy_abs_8_9_abs_le_orig: dy_abs_8_9 - dy_diff_8_9
   + 774 dy_abs_8_9_is_positive <= 774
 dy_abs_8_9_abs_ge_neg_orig: dy_abs_8_9 + dy_diff_8_9
   + 774 dy_abs_8_9_is_positive >= 0
 dy_abs_8_9_abs_le_neg_orig: dy_abs_8_9 + dy_diff_8_9
   - 774 dy_abs_8_9_is_positive <= 0
 dy_abs_8_9_force_positive: dy_diff_8_9 - 774 dy_abs_8_9_is_positive
   >= -774
 dy_abs_8_9_force_negative: dy_diff_8_9 - 774 dy_abs_8_9_is_positive <= 0
 dx_diff_def_8_12: - cx_grid_var_8 + cx_grid_var_12 + dx_diff_8_12 = 0
 dy_diff_def_8_12: - cy_grid_var_8 + cy_grid_var_12 + dy_diff_8_12 = 0
 dx_abs_8_12_abs_ge_orig: dx_abs_8_12 - dx_diff_8_12
   - 774 dx_abs_8_12_is_positive >= -774
 dx_abs_8_12_abs_le_orig: dx_abs_8_12 - dx_diff_8_12
   + 774 dx_abs_8_12_is_positive <= 774
 dx_abs_8_12_abs_ge_neg_orig: dx_abs_8_12 + dx_diff_8_12
   + 774 dx_abs_8_12_is_positive >= 0
 dx_abs_8_12_abs_le_neg_orig: dx_abs_8_12 + dx_diff_8_12
   - 774 dx_abs_8_12_is_positive <= 0
 dx_abs_8_12_force_positive: dx_diff_8_12 - 774 dx_abs_8_12_is_positive
   >= -774
 dx_abs_8_12_force_negative: dx_diff_8_12 - 774 dx_abs_8_12_is_positive
   <= 0
 dy_abs_8_12_abs_ge_orig: dy_abs_8_12 - dy_diff_8_12
   - 774 dy_abs_8_12_is_positive >= -774
 dy_abs_8_12_abs_le_orig: dy_abs_8_12 - dy_diff_8_12
   + 774 dy_abs_8_12_is_positive <= 774
 dy_abs_8_12_abs_ge_neg_orig: dy_abs_8_12 + dy_diff_8_12
   + 774 dy_abs_8_12_is_positive >= 0
 dy_abs_8_12_abs_le_neg_orig: dy_abs_8_12 + dy_diff_8_12
   - 774 dy_abs_8_12_is_positive <= 0
 dy_abs_8_12_force_positive: dy_diff_8_12 - 774 dy_abs_8_12_is_positive
   >= -774
 dy_abs_8_12_force_negative: dy_diff_8_12 - 774 dy_abs_8_12_is_positive
   <= 0
 dx_diff_def_8_13: - cx_grid_var_8 + cx_grid_var_13 + dx_diff_8_13 = 0
 dy_diff_def_8_13: - cy_grid_var_8 + cy_grid_var_13 + dy_diff_8_13 = 0
 dx_abs_8_13_abs_ge_orig: dx_abs_8_13 - dx_diff_8_13
   - 774 dx_abs_8_13_is_positive >= -774
 dx_abs_8_13_abs_le_orig: dx_abs_8_13 - dx_diff_8_13
   + 774 dx_abs_8_13_is_positive <= 774
 dx_abs_8_13_abs_ge_neg_orig: dx_abs_8_13 + dx_diff_8_13
   + 774 dx_abs_8_13_is_positive >= 0
 dx_abs_8_13_abs_le_neg_orig: dx_abs_8_13 + dx_diff_8_13
   - 774 dx_abs_8_13_is_positive <= 0
 dx_abs_8_13_force_positive: dx_diff_8_13 - 774 dx_abs_8_13_is_positive
   >= -774
 dx_abs_8_13_force_negative: dx_diff_8_13 - 774 dx_abs_8_13_is_positive
   <= 0
 dy_abs_8_13_abs_ge_orig: dy_abs_8_13 - dy_diff_8_13
   - 774 dy_abs_8_13_is_positive >= -774
 dy_abs_8_13_abs_le_orig: dy_abs_8_13 - dy_diff_8_13
   + 774 dy_abs_8_13_is_positive <= 774
 dy_abs_8_13_abs_ge_neg_orig: dy_abs_8_13 + dy_diff_8_13
   + 774 dy_abs_8_13_is_positive >= 0
 dy_abs_8_13_abs_le_neg_orig: dy_abs_8_13 + dy_diff_8_13
   - 774 dy_abs_8_13_is_positive <= 0
 dy_abs_8_13_force_positive: dy_diff_8_13 - 774 dy_abs_8_13_is_positive
   >= -774
 dy_abs_8_13_force_negative: dy_diff_8_13 - 774 dy_abs_8_13_is_positive
   <= 0
 dx_diff_def_9_10: - cx_grid_var_9 + cx_grid_var_10 + dx_diff_9_10 = 0
 dy_diff_def_9_10: - cy_grid_var_9 + cy_grid_var_10 + dy_diff_9_10 = 0
 dx_abs_9_10_abs_ge_orig: dx_abs_9_10 - dx_diff_9_10
   - 774 dx_abs_9_10_is_positive >= -774
 dx_abs_9_10_abs_le_orig: dx_abs_9_10 - dx_diff_9_10
   + 774 dx_abs_9_10_is_positive <= 774
 dx_abs_9_10_abs_ge_neg_orig: dx_abs_9_10 + dx_diff_9_10
   + 774 dx_abs_9_10_is_positive >= 0
 dx_abs_9_10_abs_le_neg_orig: dx_abs_9_10 + dx_diff_9_10
   - 774 dx_abs_9_10_is_positive <= 0
 dx_abs_9_10_force_positive: dx_diff_9_10 - 774 dx_abs_9_10_is_positive
   >= -774
 dx_abs_9_10_force_negative: dx_diff_9_10 - 774 dx_abs_9_10_is_positive
   <= 0
 dy_abs_9_10_abs_ge_orig: dy_abs_9_10 - dy_diff_9_10
   - 774 dy_abs_9_10_is_positive >= -774
 dy_abs_9_10_abs_le_orig: dy_abs_9_10 - dy_diff_9_10
   + 774 dy_abs_9_10_is_positive <= 774
 dy_abs_9_10_abs_ge_neg_orig: dy_abs_9_10 + dy_diff_9_10
   + 774 dy_abs_9_10_is_positive >= 0
 dy_abs_9_10_abs_le_neg_orig: dy_abs_9_10 + dy_diff_9_10
   - 774 dy_abs_9_10_is_positive <= 0
 dy_abs_9_10_force_positive: dy_diff_9_10 - 774 dy_abs_9_10_is_positive
   >= -774
 dy_abs_9_10_force_negative: dy_diff_9_10 - 774 dy_abs_9_10_is_positive
   <= 0
 dx_diff_def_9_11: - cx_grid_var_9 + cx_grid_var_11 + dx_diff_9_11 = 0
 dy_diff_def_9_11: - cy_grid_var_9 + cy_grid_var_11 + dy_diff_9_11 = 0
 dx_abs_9_11_abs_ge_orig: dx_abs_9_11 - dx_diff_9_11
   - 774 dx_abs_9_11_is_positive >= -774
 dx_abs_9_11_abs_le_orig: dx_abs_9_11 - dx_diff_9_11
   + 774 dx_abs_9_11_is_positive <= 774
 dx_abs_9_11_abs_ge_neg_orig: dx_abs_9_11 + dx_diff_9_11
   + 774 dx_abs_9_11_is_positive >= 0
 dx_abs_9_11_abs_le_neg_orig: dx_abs_9_11 + dx_diff_9_11
   - 774 dx_abs_9_11_is_positive <= 0
 dx_abs_9_11_force_positive: dx_diff_9_11 - 774 dx_abs_9_11_is_positive
   >= -774
 dx_abs_9_11_force_negative: dx_diff_9_11 - 774 dx_abs_9_11_is_positive
   <= 0
 dy_abs_9_11_abs_ge_orig: dy_abs_9_11 - dy_diff_9_11
   - 774 dy_abs_9_11_is_positive >= -774
 dy_abs_9_11_abs_le_orig: dy_abs_9_11 - dy_diff_9_11
   + 774 dy_abs_9_11_is_positive <= 774
 dy_abs_9_11_abs_ge_neg_orig: dy_abs_9_11 + dy_diff_9_11
   + 774 dy_abs_9_11_is_positive >= 0
 dy_abs_9_11_abs_le_neg_orig: dy_abs_9_11 + dy_diff_9_11
   - 774 dy_abs_9_11_is_positive <= 0
 dy_abs_9_11_force_positive: dy_diff_9_11 - 774 dy_abs_9_11_is_positive
   >= -774
 dy_abs_9_11_force_negative: dy_diff_9_11 - 774 dy_abs_9_11_is_positive
   <= 0
 dx_diff_def_9_14: - cx_grid_var_9 + cx_grid_var_14 + dx_diff_9_14 = 0
 dy_diff_def_9_14: - cy_grid_var_9 + cy_grid_var_14 + dy_diff_9_14 = 0
 dx_abs_9_14_abs_ge_orig: dx_abs_9_14 - dx_diff_9_14
   - 774 dx_abs_9_14_is_positive >= -774
 dx_abs_9_14_abs_le_orig: dx_abs_9_14 - dx_diff_9_14
   + 774 dx_abs_9_14_is_positive <= 774
 dx_abs_9_14_abs_ge_neg_orig: dx_abs_9_14 + dx_diff_9_14
   + 774 dx_abs_9_14_is_positive >= 0
 dx_abs_9_14_abs_le_neg_orig: dx_abs_9_14 + dx_diff_9_14
   - 774 dx_abs_9_14_is_positive <= 0
 dx_abs_9_14_force_positive: dx_diff_9_14 - 774 dx_abs_9_14_is_positive
   >= -774
 dx_abs_9_14_force_negative: dx_diff_9_14 - 774 dx_abs_9_14_is_positive
   <= 0
 dy_abs_9_14_abs_ge_orig: dy_abs_9_14 - dy_diff_9_14
   - 774 dy_abs_9_14_is_positive >= -774
 dy_abs_9_14_abs_le_orig: dy_abs_9_14 - dy_diff_9_14
   + 774 dy_abs_9_14_is_positive <= 774
 dy_abs_9_14_abs_ge_neg_orig: dy_abs_9_14 + dy_diff_9_14
   + 774 dy_abs_9_14_is_positive >= 0
 dy_abs_9_14_abs_le_neg_orig: dy_abs_9_14 + dy_diff_9_14
   - 774 dy_abs_9_14_is_positive <= 0
 dy_abs_9_14_force_positive: dy_diff_9_14 - 774 dy_abs_9_14_is_positive
   >= -774
 dy_abs_9_14_force_negative: dy_diff_9_14 - 774 dy_abs_9_14_is_positive
   <= 0
 dx_diff_def_10_11: - cx_grid_var_10 + cx_grid_var_11 + dx_diff_10_11 = 0
 dy_diff_def_10_11: - cy_grid_var_10 + cy_grid_var_11 + dy_diff_10_11 = 0
 dx_abs_10_11_abs_ge_orig: dx_abs_10_11 - dx_diff_10_11
   - 774 dx_abs_10_11_is_positive >= -774
 dx_abs_10_11_abs_le_orig: dx_abs_10_11 - dx_diff_10_11
   + 774 dx_abs_10_11_is_positive <= 774
 dx_abs_10_11_abs_ge_neg_orig: dx_abs_10_11 + dx_diff_10_11
   + 774 dx_abs_10_11_is_positive >= 0
 dx_abs_10_11_abs_le_neg_orig: dx_abs_10_11 + dx_diff_10_11
   - 774 dx_abs_10_11_is_positive <= 0
 dx_abs_10_11_force_positive: dx_diff_10_11 - 774 dx_abs_10_11_is_positive
   >= -774
 dx_abs_10_11_force_negative: dx_diff_10_11 - 774 dx_abs_10_11_is_positive
   <= 0
 dy_abs_10_11_abs_ge_orig: dy_abs_10_11 - dy_diff_10_11
   - 774 dy_abs_10_11_is_positive >= -774
 dy_abs_10_11_abs_le_orig: dy_abs_10_11 - dy_diff_10_11
   + 774 dy_abs_10_11_is_positive <= 774
 dy_abs_10_11_abs_ge_neg_orig: dy_abs_10_11 + dy_diff_10_11
   + 774 dy_abs_10_11_is_positive >= 0
 dy_abs_10_11_abs_le_neg_orig: dy_abs_10_11 + dy_diff_10_11
   - 774 dy_abs_10_11_is_positive <= 0
 dy_abs_10_11_force_positive: dy_diff_10_11 - 774 dy_abs_10_11_is_positive
   >= -774
 dy_abs_10_11_force_negative: dy_diff_10_11 - 774 dy_abs_10_11_is_positive
   <= 0
 dx_diff_def_11_15: - cx_grid_var_11 + cx_grid_var_15 + dx_diff_11_15 = 0
 dy_diff_def_11_15: - cy_grid_var_11 + cy_grid_var_15 + dy_diff_11_15 = 0
 dx_abs_11_15_abs_ge_orig: dx_abs_11_15 - dx_diff_11_15
   - 774 dx_abs_11_15_is_positive >= -774
 dx_abs_11_15_abs_le_orig: dx_abs_11_15 - dx_diff_11_15
   + 774 dx_abs_11_15_is_positive <= 774
 dx_abs_11_15_abs_ge_neg_orig: dx_abs_11_15 + dx_diff_11_15
   + 774 dx_abs_11_15_is_positive >= 0
 dx_abs_11_15_abs_le_neg_orig: dx_abs_11_15 + dx_diff_11_15
   - 774 dx_abs_11_15_is_positive <= 0
 dx_abs_11_15_force_positive: dx_diff_11_15 - 774 dx_abs_11_15_is_positive
   >= -774
 dx_abs_11_15_force_negative: dx_diff_11_15 - 774 dx_abs_11_15_is_positive
   <= 0
 dy_abs_11_15_abs_ge_orig: dy_abs_11_15 - dy_diff_11_15
   - 774 dy_abs_11_15_is_positive >= -774
 dy_abs_11_15_abs_le_orig: dy_abs_11_15 - dy_diff_11_15
   + 774 dy_abs_11_15_is_positive <= 774
 dy_abs_11_15_abs_ge_neg_orig: dy_abs_11_15 + dy_diff_11_15
   + 774 dy_abs_11_15_is_positive >= 0
 dy_abs_11_15_abs_le_neg_orig: dy_abs_11_15 + dy_diff_11_15
   - 774 dy_abs_11_15_is_positive <= 0
 dy_abs_11_15_force_positive: dy_diff_11_15 - 774 dy_abs_11_15_is_positive
   >= -774
 dy_abs_11_15_force_negative: dy_diff_11_15 - 774 dy_abs_11_15_is_positive
   <= 0
 dx_diff_def_12_13: - cx_grid_var_12 + cx_grid_var_13 + dx_diff_12_13 = 0
 dy_diff_def_12_13: - cy_grid_var_12 + cy_grid_var_13 + dy_diff_12_13 = 0
 dx_abs_12_13_abs_ge_orig: dx_abs_12_13 - dx_diff_12_13
   - 774 dx_abs_12_13_is_positive >= -774
 dx_abs_12_13_abs_le_orig: dx_abs_12_13 - dx_diff_12_13
   + 774 dx_abs_12_13_is_positive <= 774
 dx_abs_12_13_abs_ge_neg_orig: dx_abs_12_13 + dx_diff_12_13
   + 774 dx_abs_12_13_is_positive >= 0
 dx_abs_12_13_abs_le_neg_orig: dx_abs_12_13 + dx_diff_12_13
   - 774 dx_abs_12_13_is_positive <= 0
 dx_abs_12_13_force_positive: dx_diff_12_13 - 774 dx_abs_12_13_is_positive
   >= -774
 dx_abs_12_13_force_negative: dx_diff_12_13 - 774 dx_abs_12_13_is_positive
   <= 0
 dy_abs_12_13_abs_ge_orig: dy_abs_12_13 - dy_diff_12_13
   - 774 dy_abs_12_13_is_positive >= -774
 dy_abs_12_13_abs_le_orig: dy_abs_12_13 - dy_diff_12_13
   + 774 dy_abs_12_13_is_positive <= 774
 dy_abs_12_13_abs_ge_neg_orig: dy_abs_12_13 + dy_diff_12_13
   + 774 dy_abs_12_13_is_positive >= 0
 dy_abs_12_13_abs_le_neg_orig: dy_abs_12_13 + dy_diff_12_13
   - 774 dy_abs_12_13_is_positive <= 0
 dy_abs_12_13_force_positive: dy_diff_12_13 - 774 dy_abs_12_13_is_positive
   >= -774
 dy_abs_12_13_force_negative: dy_diff_12_13 - 774 dy_abs_12_13_is_positive
   <= 0
 dx_diff_def_12_15: - cx_grid_var_12 + cx_grid_var_15 + dx_diff_12_15 = 0
 dy_diff_def_12_15: - cy_grid_var_12 + cy_grid_var_15 + dy_diff_12_15 = 0
 dx_abs_12_15_abs_ge_orig: dx_abs_12_15 - dx_diff_12_15
   - 774 dx_abs_12_15_is_positive >= -774
 dx_abs_12_15_abs_le_orig: dx_abs_12_15 - dx_diff_12_15
   + 774 dx_abs_12_15_is_positive <= 774
 dx_abs_12_15_abs_ge_neg_orig: dx_abs_12_15 + dx_diff_12_15
   + 774 dx_abs_12_15_is_positive >= 0
 dx_abs_12_15_abs_le_neg_orig: dx_abs_12_15 + dx_diff_12_15
   - 774 dx_abs_12_15_is_positive <= 0
 dx_abs_12_15_force_positive: dx_diff_12_15 - 774 dx_abs_12_15_is_positive
   >= -774
 dx_abs_12_15_force_negative: dx_diff_12_15 - 774 dx_abs_12_15_is_positive
   <= 0
 dy_abs_12_15_abs_ge_orig: dy_abs_12_15 - dy_diff_12_15
   - 774 dy_abs_12_15_is_positive >= -774
 dy_abs_12_15_abs_le_orig: dy_abs_12_15 - dy_diff_12_15
   + 774 dy_abs_12_15_is_positive <= 774
 dy_abs_12_15_abs_ge_neg_orig: dy_abs_12_15 + dy_diff_12_15
   + 774 dy_abs_12_15_is_positive >= 0
 dy_abs_12_15_abs_le_neg_orig: dy_abs_12_15 + dy_diff_12_15
   - 774 dy_abs_12_15_is_positive <= 0
 dy_abs_12_15_force_positive: dy_diff_12_15 - 774 dy_abs_12_15_is_positive
   >= -774
 dy_abs_12_15_force_negative: dy_diff_12_15 - 774 dy_abs_12_15_is_positive
   <= 0
 wirelength_def: wirelength - 128 dx_abs_0_2 - 128 dy_abs_0_2
   - 512 dx_abs_0_3 - 512 dy_abs_0_3 - 1024 dx_abs_0_4 - 1024 dy_abs_0_4
   - 1024 dx_abs_0_6 - 1024 dy_abs_0_6 - 512 dx_abs_0_12 - 512 dy_abs_0_12
   - 512 dx_abs_0_14 - 512 dy_abs_0_14 - 512 dx_abs_0_15 - 512 dy_abs_0_15
   - 128 dx_abs_1_2 - 128 dy_abs_1_2 - 1024 dx_abs_1_3 - 1024 dy_abs_1_3
   - 1024 dx_abs_1_4 - 1024 dy_abs_1_4 - 128 dx_abs_1_9 - 128 dy_abs_1_9
   - 1024 dx_abs_1_12 - 1024 dy_abs_1_12 - 512 dx_abs_1_13
   - 512 dy_abs_1_13 - 256 dx_abs_1_14 - 256 dy_abs_1_14 - 256 dx_abs_1_15
   - 256 dy_abs_1_15 - 128 dx_abs_2_3 - 128 dy_abs_2_3 - 512 dx_abs_2_4
   - 512 dy_abs_2_4 - 256 dx_abs_2_6 - 256 dy_abs_2_6 - 128 dx_abs_2_7
   - 128 dy_abs_2_7 - 512 dx_abs_2_8 - 512 dy_abs_2_8 - 128 dx_abs_2_9
   - 128 dy_abs_2_9 - 1024 dx_abs_2_14 - 1024 dy_abs_2_14 - 512 dx_abs_3_4
   - 512 dy_abs_3_4 - 128 dx_abs_3_6 - 128 dy_abs_3_6 - 256 dx_abs_3_11
   - 256 dy_abs_3_11 - 256 dx_abs_3_13 - 256 dy_abs_3_13 - 256 dx_abs_3_14
   - 256 dy_abs_3_14 - 1024 dx_abs_4_6 - 1024 dy_abs_4_6 - 512 dx_abs_4_11
   - 512 dy_abs_4_11 - 512 dx_abs_4_15 - 512 dy_abs_4_15 - 1024 dx_abs_5_9
   - 1024 dy_abs_5_9 - 1024 dx_abs_5_10 - 1024 dy_abs_5_10
   - 512 dx_abs_6_10 - 512 dy_abs_6_10 - 512 dx_abs_6_12 - 512 dy_abs_6_12
   - 256 dx_abs_6_13 - 256 dy_abs_6_13 - 1024 dx_abs_6_14
   - 1024 dy_abs_6_14 - 128 dx_abs_7_11 - 128 dy_abs_7_11 - 128 dx_abs_7_12
   - 128 dy_abs_7_12 - 512 dx_abs_7_13 - 512 dy_abs_7_13 - 128 dx_abs_7_14
   - 128 dy_abs_7_14 - 512 dx_abs_8_9 - 512 dy_abs_8_9 - 1024 dx_abs_8_12
   - 1024 dy_abs_8_12 - 128 dx_abs_8_13 - 128 dy_abs_8_13 - 256 dx_abs_9_10
   - 256 dy_abs_9_10 - 128 dx_abs_9_11 - 128 dy_abs_9_11 - 1024 dx_abs_9_14
   - 1024 dy_abs_9_14 - 1024 dx_abs_10_11 - 1024 dy_abs_10_11
   - 128 dx_abs_11_15 - 128 dy_abs_11_15 - 256 dx_abs_12_13
   - 256 dy_abs_12_13 - 512 dx_abs_12_15 - 512 dy_abs_12_15 = 0
 t_ge_scaled_mean: - 0.8 bbox_w - 0.8 bbox_h + bbox_area_proxy_t >= 0
Bounds
 5 <= w_var_0 <= 6
 5 <= h_var_0 <= 6
 12 <= w_var_1 <= 15
 12 <= h_var_1 <= 15
 25 <= w_var_2 <= 26
 25 <= h_var_2 <= 26
 28 <= w_var_3 <= 29
 28 <= h_var_3 <= 29
 28 <= w_var_4 <= 29
 28 <= h_var_4 <= 29
 22 <= w_var_5 <= 23
 22 <= h_var_5 <= 23
 28 <= w_var_6 <= 29
 28 <= h_var_6 <= 29
 w_var_7 = 16
 h_var_7 = 16
 w_var_8 = 20
 h_var_8 = 20
 25 <= w_var_9 <= 27
 25 <= h_var_9 <= 27
 19 <= w_var_10 <= 20
 19 <= h_var_10 <= 20
 25 <= w_var_11 <= 28
 25 <= h_var_11 <= 28
 23 <= w_var_12 <= 25
 23 <= h_var_12 <= 25
 19 <= w_var_13 <= 22
 19 <= h_var_13 <= 22
 16 <= w_var_14 <= 19
 16 <= h_var_14 <= 19
 w_var_15 = 27
 h_var_15 = 27
 x_grid_var_0 <= 387
 y_grid_var_0 <= 387
 x_grid_var_1 <= 387
 y_grid_var_1 <= 387
 x_grid_var_2 <= 387
 y_grid_var_2 <= 387
 x_grid_var_3 <= 387
 y_grid_var_3 <= 387
 x_grid_var_4 <= 387
 y_grid_var_4 <= 387
 x_grid_var_5 <= 387
 y_grid_var_5 <= 387
 x_grid_var_6 <= 387
 y_grid_var_6 <= 387
 x_grid_var_7 <= 387
 y_grid_var_7 <= 387
 x_grid_var_8 <= 387
 y_grid_var_8 <= 387
 x_grid_var_9 <= 387
 y_grid_var_9 <= 387
 x_grid_var_10 <= 387
 y_grid_var_10 <= 387
 x_grid_var_11 <= 387
 y_grid_var_11 <= 387
 x_grid_var_12 <= 387
 y_grid_var_12 <= 387
 x_grid_var_13 <= 387
 y_grid_var_13 <= 387
 x_grid_var_14 <= 387
 y_grid_var_14 <= 387
 x_grid_var_15 <= 387
 y_grid_var_15 <= 387
 cx_grid_var_0 <= 774
 cy_grid_var_0 <= 774
 cx_grid_var_1 <= 774
 cy_grid_var_1 <= 774
 cx_grid_var_2 <= 774
 cy_grid_var_2 <= 774
 cx_grid_var_3 <= 774
 cy_grid_var_3 <= 774
 cx_grid_var_4 <= 774
 cy_grid_var_4 <= 774
 cx_grid_var_5 <= 774
 cy_grid_var_5 <= 774
 cx_grid_var_6 <= 774
 cy_grid_var_6 <= 774
 cx_grid_var_7 <= 774
 cy_grid_var_7 <= 774
 cx_grid_var_8 <= 774
 cy_grid_var_8 <= 774
 cx_grid_var_9 <= 774
 cy_grid_var_9 <= 774
 cx_grid_var_10 <= 774
 cy_grid_var_10 <= 774
 cx_grid_var_11 <= 774
 cy_grid_var_11 <= 774
 cx_grid_var_12 <= 774
 cy_grid_var_12 <= 774
 cx_grid_var_13 <= 774
 cy_grid_var_13 <= 774
 cx_grid_var_14 <= 774
 cy_grid_var_14 <= 774
 cx_grid_var_15 <= 774
 cy_grid_var_15 <= 774
 cx_center <= 387
 cy_center <= 387
 bbox_min_x <= 387
 bbox_max_x <= 387
 bbox_min_y <= 387
 bbox_max_y <= 387
 bbox_w <= 387
 bbox_h <= 387
 aspect_ratio_penalty <= 387
 wirelength <= 1.585152e+08
 -387 <= dx_diff_0_2 <= 387
 -387 <= dy_diff_0_2 <= 387
 -387 <= dx_diff_0_3 <= 387
 -387 <= dy_diff_0_3 <= 387
 -387 <= dx_diff_0_4 <= 387
 -387 <= dy_diff_0_4 <= 387
 -387 <= dx_diff_0_6 <= 387
 -387 <= dy_diff_0_6 <= 387
 -387 <= dx_diff_0_12 <= 387
 -387 <= dy_diff_0_12 <= 387
 -387 <= dx_diff_0_14 <= 387
 -387 <= dy_diff_0_14 <= 387
 -387 <= dx_diff_0_15 <= 387
 -387 <= dy_diff_0_15 <= 387
 -387 <= dx_diff_1_2 <= 387
 -387 <= dy_diff_1_2 <= 387
 -387 <= dx_diff_1_3 <= 387
 -387 <= dy_diff_1_3 <= 387
 -387 <= dx_diff_1_4 <= 387
 -387 <= dy_diff_1_4 <= 387
 -387 <= dx_diff_1_9 <= 387
 -387 <= dy_diff_1_9 <= 387
 -387 <= dx_diff_1_12 <= 387
 -387 <= dy_diff_1_12 <= 387
 -387 <= dx_diff_1_13 <= 387
 -387 <= dy_diff_1_13 <= 387
 -387 <= dx_diff_1_14 <= 387
 -387 <= dy_diff_1_14 <= 387
 -387 <= dx_diff_1_15 <= 387
 -387 <= dy_diff_1_15 <= 387
 -387 <= dx_diff_2_3 <= 387
 -387 <= dy_diff_2_3 <= 387
 -387 <= dx_diff_2_4 <= 387
 -387 <= dy_diff_2_4 <= 387
 -387 <= dx_diff_2_6 <= 387
 -387 <= dy_diff_2_6 <= 387
 -387 <= dx_diff_2_7 <= 387
 -387 <= dy_diff_2_7 <= 387
 -387 <= dx_diff_2_8 <= 387
 -387 <= dy_diff_2_8 <= 387
 -387 <= dx_diff_2_9 <= 387
 -387 <= dy_diff_2_9 <= 387
 -387 <= dx_diff_2_14 <= 387
 -387 <= dy_diff_2_14 <= 387
 -387 <= dx_diff_3_4 <= 387
 -387 <= dy_diff_3_4 <= 387
 -387 <= dx_diff_3_6 <= 387
 -387 <= dy_diff_3_6 <= 387
 -387 <= dx_diff_3_11 <= 387
 -387 <= dy_diff_3_11 <= 387
 -387 <= dx_diff_3_13 <= 387
 -387 <= dy_diff_3_13 <= 387
 -387 <= dx_diff_3_14 <= 387
 -387 <= dy_diff_3_14 <= 387
 -387 <= dx_diff_4_6 <= 387
 -387 <= dy_diff_4_6 <= 387
 -387 <= dx_diff_4_11 <= 387
 -387 <= dy_diff_4_11 <= 387
 -387 <= dx_diff_4_15 <= 387
 -387 <= dy_diff_4_15 <= 387
 -387 <= dx_diff_5_9 <= 387
 -387 <= dy_diff_5_9 <= 387
 -387 <= dx_diff_5_10 <= 387
 -387 <= dy_diff_5_10 <= 387
 -387 <= dx_diff_6_10 <= 387
 -387 <= dy_diff_6_10 <= 387
 -387 <= dx_diff_6_12 <= 387
 -387 <= dy_diff_6_12 <= 387
 -387 <= dx_diff_6_13 <= 387
 -387 <= dy_diff_6_13 <= 387
 -387 <= dx_diff_6_14 <= 387
 -387 <= dy_diff_6_14 <= 387
 -387 <= dx_diff_7_11 <= 387
 -387 <= dy_diff_7_11 <= 387
 -387 <= dx_diff_7_12 <= 387
 -387 <= dy_diff_7_12 <= 387
 -387 <= dx_diff_7_13 <= 387
 -387 <= dy_diff_7_13 <= 387
 -387 <= dx_diff_7_14 <= 387
 -387 <= dy_diff_7_14 <= 387
 -387 <= dx_diff_8_9 <= 387
 -387 <= dy_diff_8_9 <= 387
 -387 <= dx_diff_8_12 <= 387
 -387 <= dy_diff_8_12 <= 387
 -387 <= dx_diff_8_13 <= 387
 -387 <= dy_diff_8_13 <= 387
 -387 <= dx_diff_9_10 <= 387
 -387 <= dy_diff_9_10 <= 387
 -387 <= dx_diff_9_11 <= 387
 -387 <= dy_diff_9_11 <= 387
 -387 <= dx_diff_9_14 <= 387
 -387 <= dy_diff_9_14 <= 387
 -387 <= dx_diff_10_11 <= 387
 -387 <= dy_diff_10_11 <= 387
 -387 <= dx_diff_11_15 <= 387
 -387 <= dy_diff_11_15 <= 387
 -387 <= dx_diff_12_13 <= 387
 -387 <= dy_diff_12_13 <= 387
 -387 <= dx_diff_12_15 <= 387
 -387 <= dy_diff_12_15 <= 387
 bbox_area_proxy_t <= 774
Binaries
 r_0 r_1 r_2 r_3 r_4 r_5 r_6 r_7 r_8 r_9 r_10 r_11 r_12 r_13 r_14 r_15
 p_left_0_1 p_right_0_1 p_down_0_1 p_up_0_1 p_left_0_2 p_right_0_2
 p_down_0_2 p_up_0_2 p_left_0_3 p_right_0_3 p_down_0_3 p_up_0_3 p_left_0_4
 p_right_0_4 p_down_0_4 p_up_0_4 p_left_0_5 p_right_0_5 p_down_0_5 p_up_0_5
 p_left_0_6 p_right_0_6 p_down_0_6 p_up_0_6 p_left_0_7 p_right_0_7
 p_down_0_7 p_up_0_7 p_left_0_8 p_right_0_8 p_down_0_8 p_up_0_8 p_left_0_9
 p_right_0_9 p_down_0_9 p_up_0_9 p_left_0_10 p_right_0_10 p_down_0_10
 p_up_0_10 p_left_0_11 p_right_0_11 p_down_0_11 p_up_0_11 p_left_0_12
 p_right_0_12 p_down_0_12 p_up_0_12 p_left_0_13 p_right_0_13 p_down_0_13
 p_up_0_13 p_left_0_14 p_right_0_14 p_down_0_14 p_up_0_14 p_left_0_15
 p_right_0_15 p_down_0_15 p_up_0_15 p_left_1_2 p_right_1_2 p_down_1_2
 p_up_1_2 p_left_1_3 p_right_1_3 p_down_1_3 p_up_1_3 p_left_1_4 p_right_1_4
 p_down_1_4 p_up_1_4 p_left_1_5 p_right_1_5 p_down_1_5 p_up_1_5 p_left_1_6
 p_right_1_6 p_down_1_6 p_up_1_6 p_left_1_7 p_right_1_7 p_down_1_7 p_up_1_7
 p_left_1_8 p_right_1_8 p_down_1_8 p_up_1_8 p_left_1_9 p_right_1_9
 p_down_1_9 p_up_1_9 p_left_1_10 p_right_1_10 p_down_1_10 p_up_1_10
 p_left_1_11 p_right_1_11 p_down_1_11 p_up_1_11 p_left_1_12 p_right_1_12
 p_down_1_12 p_up_1_12 p_left_1_13 p_right_1_13 p_down_1_13 p_up_1_13
 p_left_1_14 p_right_1_14 p_down_1_14 p_up_1_14 p_left_1_15 p_right_1_15
 p_down_1_15 p_up_1_15 p_left_2_3 p_right_2_3 p_down_2_3 p_up_2_3
 p_left_2_4 p_right_2_4 p_down_2_4 p_up_2_4 p_left_2_5 p_right_2_5
 p_down_2_5 p_up_2_5 p_left_2_6 p_right_2_6 p_down_2_6 p_up_2_6 p_left_2_7
 p_right_2_7 p_down_2_7 p_up_2_7 p_left_2_8 p_right_2_8 p_down_2_8 p_up_2_8
 p_left_2_9 p_right_2_9 p_down_2_9 p_up_2_9 p_left_2_10 p_right_2_10
 p_down_2_10 p_up_2_10 p_left_2_11 p_right_2_11 p_down_2_11 p_up_2_11
 p_left_2_12 p_right_2_12 p_down_2_12 p_up_2_12 p_left_2_13 p_right_2_13
 p_down_2_13 p_up_2_13 p_left_2_14 p_right_2_14 p_down_2_14 p_up_2_14
 p_left_2_15 p_right_2_15 p_down_2_15 p_up_2_15 p_left_3_4 p_right_3_4
 p_down_3_4 p_up_3_4 p_left_3_5 p_right_3_5 p_down_3_5 p_up_3_5 p_left_3_6
 p_right_3_6 p_down_3_6 p_up_3_6 p_left_3_7 p_right_3_7 p_down_3_7 p_up_3_7
 p_left_3_8 p_right_3_8 p_down_3_8 p_up_3_8 p_left_3_9 p_right_3_9
 p_down_3_9 p_up_3_9 p_left_3_10 p_right_3_10 p_down_3_10 p_up_3_10
 p_left_3_11 p_right_3_11 p_down_3_11 p_up_3_11 p_left_3_12 p_right_3_12
 p_down_3_12 p_up_3_12 p_left_3_13 p_right_3_13 p_down_3_13 p_up_3_13
 p_left_3_14 p_right_3_14 p_down_3_14 p_up_3_14 p_left_3_15 p_right_3_15
 p_down_3_15 p_up_3_15 p_left_4_5 p_right_4_5 p_down_4_5 p_up_4_5
 p_left_4_6 p_right_4_6 p_down_4_6 p_up_4_6 p_left_4_7 p_right_4_7
 p_down_4_7 p_up_4_7 p_left_4_8 p_right_4_8 p_down_4_8 p_up_4_8 p_left_4_9
 p_right_4_9 p_down_4_9 p_up_4_9 p_left_4_10 p_right_4_10 p_down_4_10
 p_up_4_10 p_left_4_11 p_right_4_11 p_down_4_11 p_up_4_11 p_left_4_12
 p_right_4_12 p_down_4_12 p_up_4_12 p_left_4_13 p_right_4_13 p_down_4_13
 p_up_4_13 p_left_4_14 p_right_4_14 p_down_4_14 p_up_4_14 p_left_4_15
 p_right_4_15 p_down_4_15 p_up_4_15 p_left_5_6 p_right_5_6 p_down_5_6
 p_up_5_6 p_left_5_7 p_right_5_7 p_down_5_7 p_up_5_7 p_left_5_8 p_right_5_8
 p_down_5_8 p_up_5_8 p_left_5_9 p_right_5_9 p_down_5_9 p_up_5_9 p_left_5_10
 p_right_5_10 p_down_5_10 p_up_5_10 p_left_5_11 p_right_5_11 p_down_5_11
 p_up_5_11 p_left_5_12 p_right_5_12 p_down_5_12 p_up_5_12 p_left_5_13
 p_right_5_13 p_down_5_13 p_up_5_13 p_left_5_14 p_right_5_14 p_down_5_14
 p_up_5_14 p_left_5_15 p_right_5_15 p_down_5_15 p_up_5_15 p_left_6_7
 p_right_6_7 p_down_6_7 p_up_6_7 p_left_6_8 p_right_6_8 p_down_6_8 p_up_6_8
 p_left_6_9 p_right_6_9 p_down_6_9 p_up_6_9 p_left_6_10 p_right_6_10
 p_down_6_10 p_up_6_10 p_left_6_11 p_right_6_11 p_down_6_11 p_up_6_11
 p_left_6_12 p_right_6_12 p_down_6_12 p_up_6_12 p_left_6_13 p_right_6_13
 p_down_6_13 p_up_6_13 p_left_6_14 p_right_6_14 p_down_6_14 p_up_6_14
 p_left_6_15 p_right_6_15 p_down_6_15 p_up_6_15 p_left_7_8 p_right_7_8
 p_down_7_8 p_up_7_8 p_left_7_9 p_right_7_9 p_down_7_9 p_up_7_9 p_left_7_10
 p_right_7_10 p_down_7_10 p_up_7_10 p_left_7_11 p_right_7_11 p_down_7_11
 p_up_7_11 p_left_7_12 p_right_7_12 p_down_7_12 p_up_7_12 p_left_7_13
 p_right_7_13 p_down_7_13 p_up_7_13 p_left_7_14 p_right_7_14 p_down_7_14
 p_up_7_14 p_left_7_15 p_right_7_15 p_down_7_15 p_up_7_15 p_left_8_9
 p_right_8_9 p_down_8_9 p_up_8_9 p_left_8_10 p_right_8_10 p_down_8_10
 p_up_8_10 p_left_8_11 p_right_8_11 p_down_8_11 p_up_8_11 p_left_8_12
 p_right_8_12 p_down_8_12 p_up_8_12 p_left_8_13 p_right_8_13 p_down_8_13
 p_up_8_13 p_left_8_14 p_right_8_14 p_down_8_14 p_up_8_14 p_left_8_15
 p_right_8_15 p_down_8_15 p_up_8_15 p_left_9_10 p_right_9_10 p_down_9_10
 p_up_9_10 p_left_9_11 p_right_9_11 p_down_9_11 p_up_9_11 p_left_9_12
 p_right_9_12 p_down_9_12 p_up_9_12 p_left_9_13 p_right_9_13 p_down_9_13
 p_up_9_13 p_left_9_14 p_right_9_14 p_down_9_14 p_up_9_14 p_left_9_15
 p_right_9_15 p_down_9_15 p_up_9_15 p_left_10_11 p_right_10_11 p_down_10_11
 p_up_10_11 p_left_10_12 p_right_10_12 p_down_10_12 p_up_10_12 p_left_10_13
 p_right_10_13 p_down_10_13 p_up_10_13 p_left_10_14 p_right_10_14
 p_down_10_14 p_up_10_14 p_left_10_15 p_right_10_15 p_down_10_15 p_up_10_15
 p_left_11_12 p_right_11_12 p_down_11_12 p_up_11_12 p_left_11_13
 p_right_11_13 p_down_11_13 p_up_11_13 p_left_11_14 p_right_11_14
 p_down_11_14 p_up_11_14 p_left_11_15 p_right_11_15 p_down_11_15 p_up_11_15
 p_left_12_13 p_right_12_13 p_down_12_13 p_up_12_13 p_left_12_14
 p_right_12_14 p_down_12_14 p_up_12_14 p_left_12_15 p_right_12_15
 p_down_12_15 p_up_12_15 p_left_13_14 p_right_13_14 p_down_13_14 p_up_13_14
 p_left_13_15 p_right_13_15 p_down_13_15 p_up_13_15 p_left_14_15
 p_right_14_15 p_down_14_15 p_up_14_15 dx_abs_0_2_is_positive
 dy_abs_0_2_is_positive dx_abs_0_3_is_positive dy_abs_0_3_is_positive
 dx_abs_0_4_is_positive dy_abs_0_4_is_positive dx_abs_0_6_is_positive
 dy_abs_0_6_is_positive dx_abs_0_12_is_positive dy_abs_0_12_is_positive
 dx_abs_0_14_is_positive dy_abs_0_14_is_positive dx_abs_0_15_is_positive
 dy_abs_0_15_is_positive dx_abs_1_2_is_positive dy_abs_1_2_is_positive
 dx_abs_1_3_is_positive dy_abs_1_3_is_positive dx_abs_1_4_is_positive
 dy_abs_1_4_is_positive dx_abs_1_9_is_positive dy_abs_1_9_is_positive
 dx_abs_1_12_is_positive dy_abs_1_12_is_positive dx_abs_1_13_is_positive
 dy_abs_1_13_is_positive dx_abs_1_14_is_positive dy_abs_1_14_is_positive
 dx_abs_1_15_is_positive dy_abs_1_15_is_positive dx_abs_2_3_is_positive
 dy_abs_2_3_is_positive dx_abs_2_4_is_positive dy_abs_2_4_is_positive
 dx_abs_2_6_is_positive dy_abs_2_6_is_positive dx_abs_2_7_is_positive
 dy_abs_2_7_is_positive dx_abs_2_8_is_positive dy_abs_2_8_is_positive
 dx_abs_2_9_is_positive dy_abs_2_9_is_positive dx_abs_2_14_is_positive
 dy_abs_2_14_is_positive dx_abs_3_4_is_positive dy_abs_3_4_is_positive
 dx_abs_3_6_is_positive dy_abs_3_6_is_positive dx_abs_3_11_is_positive
 dy_abs_3_11_is_positive dx_abs_3_13_is_positive dy_abs_3_13_is_positive
 dx_abs_3_14_is_positive dy_abs_3_14_is_positive dx_abs_4_6_is_positive
 dy_abs_4_6_is_positive dx_abs_4_11_is_positive dy_abs_4_11_is_positive
 dx_abs_4_15_is_positive dy_abs_4_15_is_positive dx_abs_5_9_is_positive
 dy_abs_5_9_is_positive dx_abs_5_10_is_positive dy_abs_5_10_is_positive
 dx_abs_6_10_is_positive dy_abs_6_10_is_positive dx_abs_6_12_is_positive
 dy_abs_6_12_is_positive dx_abs_6_13_is_positive dy_abs_6_13_is_positive
 dx_abs_6_14_is_positive dy_abs_6_14_is_positive dx_abs_7_11_is_positive
 dy_abs_7_11_is_positive dx_abs_7_12_is_positive dy_abs_7_12_is_positive
 dx_abs_7_13_is_positive dy_abs_7_13_is_positive dx_abs_7_14_is_positive
 dy_abs_7_14_is_positive dx_abs_8_9_is_positive dy_abs_8_9_is_positive
 dx_abs_8_12_is_positive dy_abs_8_12_is_positive dx_abs_8_13_is_positive
 dy_abs_8_13_is_positive dx_abs_9_10_is_positive dy_abs_9_10_is_positive
 dx_abs_9_11_is_positive dy_abs_9_11_is_positive dx_abs_9_14_is_positive
 dy_abs_9_14_is_positive dx_abs_10_11_is_positive dy_abs_10_11_is_positive
 dx_abs_11_15_is_positive dy_abs_11_15_is_positive dx_abs_12_13_is_positive
 dy_abs_12_13_is_positive dx_abs_12_15_is_positive dy_abs_12_15_is_positive
End
