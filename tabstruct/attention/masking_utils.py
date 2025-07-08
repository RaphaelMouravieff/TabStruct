import torch 



def gen_mask(token_type, name) :

    segment_ids = token_type[:,:, 0]
    column_ids = token_type[:,:, 1]
    row_ids = token_type[:,:, 2]

    cell_mask = (row_ids != 0) & (segment_ids == 1)
    header_mask = (row_ids == 0) & (segment_ids == 1)
    
    if name.startswith("STM") or name.startswith("SSTM"):
        special_ids = token_type[:, :, 3]
        special_token_sep_mask = (special_ids==1) & (segment_ids == 1) 
        special_token_row_mask = (special_ids==2) & (segment_ids == 1) 
        special_token_col_mask = (special_ids==3) & (segment_ids == 1) 
        special_token_cells_mask = (special_ids==4) & (segment_ids == 1) 

    if name == "mask_query_table_1":
        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 0
        mask0 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))

        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 1
        mask1 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))
        mask = (mask0 | mask1)

    if name == "mask_query_table_2":
        lines_tensor_expanded = segment_ids.unsqueeze(1)

        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 0
        mask0 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))
        mask = ~(mask0 < lines_tensor_expanded)

    if name == "mask_query_table_3":
        lines_tensor_expanded = segment_ids.unsqueeze(1)

        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 0
        mask0 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))
        mask = ~(mask0 < lines_tensor_expanded.transpose(1, 2))


    if name == 'M_query': 
        segment_zero = segment_ids == 0
        mask =  segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1)
        
    if name == "M_self": 
        device = token_type.device
        seq_len = token_type.size(1)
        arange_tensor = torch.arange(1, seq_len + 1, device=device)
        mask = (arange_tensor.unsqueeze(0).unsqueeze(1) == arange_tensor.unsqueeze(0).unsqueeze(2))

    if name == "M_cells":
        mask8 = column_ids.unsqueeze(1) == column_ids.unsqueeze(2)
        mask9 = row_ids.unsqueeze(1) == row_ids.unsqueeze(2)
        mask = mask8 & mask9
        
    if name == "M_columns":
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)))
        
    if name == "M_rows":
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)))
        
    if name == "STM_1": # Special Token Row -> Row
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  ~(special_token_row_mask.unsqueeze(1) == special_token_row_mask.unsqueeze(2)))
    
    if name == "STM_2": # Special Token Cell -> Cell
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2)))
    
    if name == "STM_3": # Special Token Col -> Col
        cell_to_header_mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                              header_mask.unsqueeze(1) &
                              cell_mask.unsqueeze(2) & 
                              ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        special_col_attention = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                                cell_mask.unsqueeze(1) &
                                header_mask.unsqueeze(2) &
                                ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        mask = torch.logical_or(special_col_attention, cell_to_header_mask)


    if name == "STM_4": # Special Token table -> table
        mask = (~(special_token_sep_mask.unsqueeze(1) == special_token_sep_mask.unsqueeze(2))&
               (segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)))

    if name == "STM_5": # Special token cells -> Special token cells
        mask = ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))
        
        
        
        
    if name == "SSTM_1":
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                   cell_mask.unsqueeze(1) &
                   cell_mask.unsqueeze(2) & 
                   ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))&
                   ~(special_token_row_mask.unsqueeze(1) == special_token_row_mask.unsqueeze(2)))

    if name == "SSTM_2":
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2)))
    
    if name == "SSTM_3":
        cell_to_header_mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                                header_mask.unsqueeze(1) &
                                cell_mask.unsqueeze(2) & 
                                ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))&
                                ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        special_col_attention = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                                 cell_mask.unsqueeze(1) &
                                 header_mask.unsqueeze(2) &
                                 ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))&
                                 ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        mask = torch.logical_or(special_col_attention, cell_to_header_mask)
        
    if name == "SSTM_4":
        mask = (~(special_token_sep_mask.unsqueeze(1) == special_token_sep_mask.unsqueeze(2))&
               (~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))|
                ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)) |
                ~(special_token_row_mask.unsqueeze(1) == special_token_row_mask.unsqueeze(2)) ) &
                (segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)))

        
    return mask


import torch 



def gen_mask(token_type, name) :

    segment_ids = token_type[:,:, 0]
    column_ids = token_type[:,:, 1]
    row_ids = token_type[:,:, 2]

    cell_mask = (row_ids != 0) & (segment_ids == 1)
    header_mask = (row_ids == 0) & (segment_ids == 1)
    
    if name.startswith("STM") or name.startswith("SSTM"):
        special_ids = token_type[:, :, 3]
        special_token_sep_mask = (special_ids==1) & (segment_ids == 1) 
        special_token_row_mask = (special_ids==2) & (segment_ids == 1) 
        special_token_col_mask = (special_ids==3) & (segment_ids == 1) 
        special_token_cells_mask = (special_ids==4) & (segment_ids == 1) 

    if name == "mask_query_table_1":
        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 0
        mask0 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))

        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 1
        mask1 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))
        mask = (mask0 | mask1)

    if name == "mask_query_table_2":
        lines_tensor_expanded = segment_ids.unsqueeze(1)

        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 0
        mask0 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))
        mask = ~(mask0 < lines_tensor_expanded)

    if name == "mask_query_table_3":
        lines_tensor_expanded = segment_ids.unsqueeze(1)

        segment_ids = token_type[:,:, 0]
        segment_zero = segment_ids == 0
        mask0 =  ~(segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1))
        mask = ~(mask0 < lines_tensor_expanded.transpose(1, 2))


    if name == 'M_query': 
        segment_zero = segment_ids == 0
        mask =  segment_zero.unsqueeze(2) | segment_zero.unsqueeze(1)
        
    if name == "M_self": 
        device = token_type.device
        seq_len = token_type.size(1)
        arange_tensor = torch.arange(1, seq_len + 1, device=device)
        mask = (arange_tensor.unsqueeze(0).unsqueeze(1) == arange_tensor.unsqueeze(0).unsqueeze(2))

    if name == "M_cells":
        mask8 = column_ids.unsqueeze(1) == column_ids.unsqueeze(2)
        mask9 = row_ids.unsqueeze(1) == row_ids.unsqueeze(2)
        mask = mask8 & mask9
        
    if name == "M_columns":
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)))
        
    if name == "M_rows":
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)))
        
    if name == "STM_1": # Special Token Row -> Row
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  ~(special_token_row_mask.unsqueeze(1) == special_token_row_mask.unsqueeze(2)))
    
    if name == "STM_2": # Special Token Cell -> Cell
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2)))
    
    if name == "STM_3": # Special Token Col -> Col
        cell_to_header_mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                              header_mask.unsqueeze(1) &
                              cell_mask.unsqueeze(2) & 
                              ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        special_col_attention = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                                cell_mask.unsqueeze(1) &
                                header_mask.unsqueeze(2) &
                                ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        mask = torch.logical_or(special_col_attention, cell_to_header_mask)


    if name == "STM_4": # Special Token table -> table
        mask = (~(special_token_sep_mask.unsqueeze(1) == special_token_sep_mask.unsqueeze(2))&
               (segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)))

    if name == "STM_5": # Special token cells -> Special token cells
        mask = ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))
        
        
        
        
    if name == "SSTM_1":
        mask = ((row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                   cell_mask.unsqueeze(1) &
                   cell_mask.unsqueeze(2) & 
                   ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))&
                   ~(special_token_row_mask.unsqueeze(1) == special_token_row_mask.unsqueeze(2)))

    if name == "SSTM_2":
        mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                  cell_mask.unsqueeze(1) &
                  cell_mask.unsqueeze(2) & 
                  (row_ids.unsqueeze(1) == row_ids.unsqueeze(2)) &
                  ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2)))
    
    if name == "SSTM_3":
        cell_to_header_mask = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                                header_mask.unsqueeze(1) &
                                cell_mask.unsqueeze(2) & 
                                ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))&
                                ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        special_col_attention = ((column_ids.unsqueeze(1) == column_ids.unsqueeze(2)) &
                                 cell_mask.unsqueeze(1) &
                                 header_mask.unsqueeze(2) &
                                 ~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))&
                                 ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)))
        mask = torch.logical_or(special_col_attention, cell_to_header_mask)
        
    if name == "SSTM_4":
        mask = (~(special_token_sep_mask.unsqueeze(1) == special_token_sep_mask.unsqueeze(2))&
               (~(special_token_cells_mask.unsqueeze(1) == special_token_cells_mask.unsqueeze(2))|
                ~(special_token_col_mask.unsqueeze(1) == special_token_col_mask.unsqueeze(2)) |
                ~(special_token_row_mask.unsqueeze(1) == special_token_row_mask.unsqueeze(2)) ) &
                (segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)))

        
    return mask






def generate_mask(token_type, attention_mask, mask_number, mask_query_table=0):

    assert 1 <= mask_number <= 13, "mask_number must be between 1 and 9 inclusive"
    
    attention_mask_global = (attention_mask.unsqueeze(1)==1) & ( 1 == attention_mask.unsqueeze(2)) 
        
    if mask_number==1: # M1
        mask = (  gen_mask(token_type, "M_query") |
                  gen_mask(token_type, "M_rows")  |
                  gen_mask(token_type, "M_columns") ) 
    
    elif mask_number==2: # M2
        mask = (  gen_mask(token_type, "M_query")  | 
                  gen_mask(token_type, "M_columns"))
        
    elif mask_number==3: # M4 -> M3 
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "M_rows"))

    if mask_number==4: # M9 -> M4
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "STM_1")  | 
                  gen_mask(token_type, "STM_2")  |
                  gen_mask(token_type, "STM_3")  |
                  gen_mask(token_type, "STM_4")  |
                  gen_mask(token_type, "M_self") |
                  gen_mask(token_type, "M_columns"))

    elif mask_number==5: # M3 -> M5
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "STM_1")  | 
                  gen_mask(token_type, "STM_2")  |
                  gen_mask(token_type, "STM_3")  |
                  gen_mask(token_type, "STM_4")  |
                  gen_mask(token_type, "M_self") |
                  gen_mask(token_type, "M_rows"))

    elif mask_number==6: # M5 -> M6
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "STM_1")  | 
                  gen_mask(token_type, "STM_2")  |
                  gen_mask(token_type, "STM_3")  |
                  gen_mask(token_type, "STM_4")  |
                  gen_mask(token_type, "M_self") |
                  gen_mask(token_type, "M_cells"))
        
    if mask_number==7: # M6 -> M7
        mask = (  gen_mask(token_type, "M_query")| 
                  gen_mask(token_type, "STM_1")  | 
                  gen_mask(token_type, "STM_2")  |
                  gen_mask(token_type, "STM_3")  |
                  gen_mask(token_type, "STM_4")  |
                  gen_mask(token_type, "M_self"))   

    if mask_number==8: # M7 -> M8
        mask = (  gen_mask(token_type, "M_query") | 
                  gen_mask(token_type, "SSTM_1")  | 
                  gen_mask(token_type, "SSTM_2")  |
                  gen_mask(token_type, "SSTM_3")  |
                  gen_mask(token_type, "SSTM_4")  | 
                  gen_mask(token_type, "M_self"))
        

    if mask_number == 10:  # M10: M1 with causal masking
        seq_len = token_type.size(1)
        causal_mask = torch.arange(seq_len, device=token_type.device).unsqueeze(1) >= \
                    torch.arange(seq_len, device=token_type.device).unsqueeze(0)
        mask = (gen_mask(token_type, "M_query") |
                gen_mask(token_type, "M_rows") |
                gen_mask(token_type, "M_columns")) & causal_mask

    if mask_number == 11:  # M11: M1 with causal masking except for the query part
        seq_len = token_type.size(1)
        causal_mask = torch.arange(seq_len, device=token_type.device).unsqueeze(1) >= \
                    torch.arange(seq_len, device=token_type.device).unsqueeze(0)
        mask = (gen_mask(token_type, "M_query") |
                ((gen_mask(token_type, "M_rows") |
                gen_mask(token_type, "M_columns")) & causal_mask))

    if mask_number == 12:  # M10: M1 with causal masking
        seq_len = token_type.size(1)
        causal_mask = torch.arange(seq_len, device=token_type.device).unsqueeze(1) >= \
                    torch.arange(seq_len, device=token_type.device).unsqueeze(0)
        mask = attention_mask_global & causal_mask

    if mask_number == 13:  # M11: M1 with causal masking except for the query part
        seq_len = token_type.size(1)
        causal_mask = torch.arange(seq_len, device=token_type.device).unsqueeze(1) >= \
                    torch.arange(seq_len, device=token_type.device).unsqueeze(0)
        mask = (attention_mask_global & causal_mask) | gen_mask(token_type, "M_query") 


        
    if mask_query_table==1:
        mask_ = gen_mask(token_type, "mask_query_table_1")
        mask = mask & mask_

    if mask_query_table==2:
        mask_ = gen_mask(token_type, "mask_query_table_2")
        mask = mask & mask_

    if mask_query_table==3:
        mask_ = gen_mask(token_type, "mask_query_table_3")
        mask = mask & mask_

    attention_mask = (mask & attention_mask_global) 

    inverted_mask = 1.0 - attention_mask.float()
    inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inverted_mask.dtype).min)

    return inverted_mask