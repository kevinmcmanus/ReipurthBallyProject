from astropy.table import Table, join

def cross_match(left_table, right_table, xmatch_table,
                keys = {'left_table':{'left_colkey':'source_id','left_xmkey':'typed_id'},
                        'right_table':{'right_colkey':'objID', 'right_xmkey': 'original_ext_source_id'}}):

    left_xmkey = keys['left_table']['left_xmkey']
    left_colkey = keys['left_table']['left_colkey']
    right_xmkey =keys['right_table']['right_xmkey']
    right_colkey = keys['right_table']['right_colkey']

    #create local xmatch table and deal with missing key values:
    xmatch = xmatch_table[[left_xmkey, right_xmkey]][~xmatch_table[right_xmkey].mask].copy()
    #fix up the column names to match to the tables to be joined
    xmatch[left_xmkey].name = left_colkey
    xmatch[right_xmkey].name = right_colkey
    xmatch.add_index(right_colkey)

    #put the match table on the right table:
    tbl_right = join(right_table, xmatch, keys=right_colkey, join_type='left')

    #join the right table to the left
    tbl_left = join(left_table, tbl_right, keys=left_colkey, join_type='left')
    tbl_left.add_index(left_colkey)

    return tbl_left
