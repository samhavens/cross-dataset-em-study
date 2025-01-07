from collections import defaultdict

from pandas import merge
import py_entitymatching as em

def verify_blocking_ground_truth(A, B, block_df, duplicates_df, objectify=False):
    num_duplicates_missed = 0
    duplicates_df.columns = ["ltable_id", "rtable_id"]
    # Sometimes pandas / Magellan puts some columns as objects instead of numeric/string. In this case, we will force this to join appropriately
    if objectify:
        duplicates_df = duplicates_df.astype(object)

    # Intuition: merge function joints two data frames. The outer option creates a number of NaN rows when
    # some duplicates are missing in the blocked_df
    # we leverage the fact that len gives all rows while count gives non-NaN to compute the missing options
    merged_df = block_df.merge(duplicates_df, left_on=["ltable_id", "rtable_id"], right_on=["ltable_id", "rtable_id"],
                               how='outer')
    num_duplicates_missed = len(merged_df) - merged_df["_id"].count()
    total_duplicates = len(duplicates_df)

    print("Ratio saved=", 1.0 - float(len(block_df)) / float(len(A) * len(B)))
    print("Totally missed:", num_duplicates_missed, " out of ", total_duplicates)


def blocking_for_abt(A,B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, "name", "name", word_level=True, overlap_size=1,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=True, allow_missing=True)
    return C


def blocking_for_amgo(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, "title", "title", word_level=True, overlap_size=1,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=False, allow_missing=True)
    return C


def blocking_for_beer(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, "Beer_Name", "Beer_Name", word_level=True, overlap_size=1,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=False, allow_missing=True)
    return C


def blocking_for_dbac(A, B):
    ab = em.AttrEquivalenceBlocker()
    attributes = A.columns.tolist()
    C = ab.block_tables(A, B, l_block_attr='year', r_block_attr='year',
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        allow_missing=True)
    ob = em.OverlapBlocker()
    C2 = ob.block_candset(C, 'title', 'title', word_level=True, overlap_size=2,
                          show_progress=False, allow_missing=True)
    return C2


def blocking_for_dbgo(A, B):
    ab = em.AttrEquivalenceBlocker()
    attributes = A.columns.tolist()
    C1 = ab.block_tables(A, B, l_block_attr='year', r_block_attr='year',
                         l_output_attrs=attributes, r_output_attrs=attributes,
                         allow_missing=True)
    ob = em.OverlapBlocker()
    C2 = ob.block_candset(C1, 'title', 'title', word_level=True, overlap_size=2,
                          show_progress=False, allow_missing=True)
    return C2


def blocking_for_foza(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, 'name', 'name',
                        l_output_attrs=attributes, r_output_attrs=attributes, overlap_size=1,
                        show_progress=False, allow_missing=True)
    return C


def blocking_for_itam(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, "Song_Name", "Song_Name", word_level=True, overlap_size=1,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=False, allow_missing=True)
    return C


def blocking_for_roim(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, 'name', 'name', word_level=True, overlap_size=2,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=False, allow_missing=True)
    return C


def blocking_for_waam(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=3,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=False, allow_missing=True)
    return C


def blocking_for_wdc(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, 'title', 'title', word_level=True, overlap_size=2,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=False, allow_missing=True)
    return C


def blocking_for_zoye(A, B):
    ob = em.OverlapBlocker()
    attributes = A.columns.tolist()
    C = ob.block_tables(A, B, 'name', 'name', word_level=True, overlap_size=2,
                        l_output_attrs=attributes, r_output_attrs=attributes,
                        show_progress=False, allow_missing=True)
    return C


def generic_blocking_func(A, B):
    A_prefix = A.add_prefix('ltable_')
    B_prefix = B.add_prefix('rtable_')
    A_prefix['key'] = 1
    B_prefix['key'] = 1
    final = merge(A_prefix, B_prefix,on='key', suffixes=('', ''))
    final = final.drop(columns=['key'])
    final = final.reset_index()
    final = final.rename(columns={'index': '_id'})
    print (list(final))
    return final


blocking_functions_mapping = defaultdict(str)
blocking_functions_mapping["abt"] = blocking_for_abt
blocking_functions_mapping["amgo"] = blocking_for_amgo
blocking_functions_mapping["beer"] = blocking_for_beer
blocking_functions_mapping["dbac"] = blocking_for_dbac
blocking_functions_mapping["dbgo"] = blocking_for_dbgo
blocking_functions_mapping["foza"] = blocking_for_foza
blocking_functions_mapping["itam"] = blocking_for_itam
blocking_functions_mapping["roim"] = blocking_for_roim
blocking_functions_mapping["waam"] = blocking_for_waam
blocking_functions_mapping["wdc"] = blocking_for_wdc
blocking_functions_mapping["zoye"] = blocking_for_zoye