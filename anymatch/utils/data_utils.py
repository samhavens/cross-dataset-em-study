import os
import random

import pandas as pd
# from autogluon.tabular import TabularPredictor


def df_serializer(data: pd.DataFrame, mode, seed=42):
    random.seed(seed)
    attrs = [col[:-2] for col in data.columns if col.endswith('_l')]
    random.shuffle(attrs)

    attrs_l = [attr + '_l' for attr in attrs]
    attrs_r = [attr + '_r' for attr in attrs]

    if mode == 'mode1':
        template_l = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        template_r = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        data['text_l'] = data.apply(lambda x: template_l.format(*x[attrs_l].fillna('N/A')), axis=1)
        data['text_r'] = data.apply(lambda x: template_r.format(*x[attrs_r].fillna('N/A')), axis=1)
        data['text'] = data.apply(lambda x: 'Record A is <p>' + x['text_l'] + '</p>. Record B is <p>' + x[
            'text_r'] + '</p>. Given the attributes of the two records, are they the same?', axis=1)

    elif mode == 'mode2':
        template_l = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        template_r = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        data['text_l'] = data.apply(lambda x: template_l.format(*x[attrs_l].fillna('N/A')), axis=1)
        data['text_r'] = data.apply(lambda x: template_r.format(*x[attrs_r].fillna('N/A')), axis=1)
        data['text'] = data.apply(lambda x: 'Given the attributes of two records, are they the same? Record A is <p>'
                                            + x['text_l'] + '</p>. Record B is <p>' + x['text_r'] + '</p>.', axis=1)
    elif mode == 'mode3':
        template_l = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        template_r = 'COL {}, ' * (len(attrs) - 1) + 'COL {}'
        data['text_l'] = data.apply(lambda x: template_l.format(*x[attrs_l].fillna('N/A')), axis=1)
        data['text_r'] = data.apply(lambda x: template_r.format(*x[attrs_r].fillna('N/A')), axis=1)
        data['text'] = data.apply(lambda x: 'Given the attributes of two records, are they the same? Record A is '
                                            + x['text_l'] + '. Record B is ' + x['text_r'] + '.', axis=1)

    elif mode == 'mode4':
        template_l = '{}: {}, ' * (len(attrs) - 1) + '{}: {}'
        template_r = '{}: {}, ' * (len(attrs) - 1) + '{}: {}'
        data['text_l'] = data.apply(
            lambda x: template_l.format(*[item for pair in zip(attrs, x[attrs_l].fillna('N/A')) for item in pair]),
            axis=1)
        data['text_r'] = data.apply(
            lambda x: template_r.format(*[item for pair in zip(attrs, x[attrs_r].fillna('N/A')) for item in pair]),
            axis=1)
        data['text'] = data.apply(lambda x: 'Given the attributes of two records, are they the same? Record A is '
                                            + x['text_l'] + '. Record B is ' + x['text_r'] + '.', axis=1)
    else:
        raise ValueError('Invalid mode')
    return data[['text', 'label']]


def one_pos_two_neg(train_df, dataset_dir):
    if len(train_df) < 1200:
        return train_df
    else:
        train_pos_pairs = train_df[train_df['label'] == 1]
        train_neg_pairs = train_df[train_df['label'] == 0]
        train_neg_pairs_sampled = train_neg_pairs.sample(n=2*len(train_pos_pairs), random_state=42)
        train_df_sampled = pd.concat([train_pos_pairs, train_neg_pairs_sampled])
        train_num = min(1200, len(train_df_sampled))
        train_df_sampled = train_df_sampled.sample(n=train_num, random_state=42).reset_index(drop=True)
        return train_df_sampled


def automl_selection(dataset_dir, max_pos_size=400):
    automl_path = dataset_dir.replace('processed', 'processed/automl') + '/train_preds.csv'
    df = pd.read_csv(automl_path)
    pos_indices =df[df['label'] == 1].sort_values('uncertainty', ascending=False).index
    neg_indices = df[df['label'] == 0].sort_values('uncertainty', ascending=False).index
    max_pos_size = min(max_pos_size, len(pos_indices))
    max_neg_size = 2 * max_pos_size
    pos_indices_selection = pos_indices[:max_pos_size]
    neg_indices_selection = neg_indices[:max_neg_size]
    return pos_indices_selection, neg_indices_selection


def automl_filter(train_df, dataset_dir):
    if len(train_df) < 1200:
        return train_df
    else:
        pos_indices_selection, neg_indices_selection = automl_selection(dataset_dir)
        indices_selction = pos_indices_selection.union(neg_indices_selection)
        filtered_train_df = train_df.loc[indices_selction].reset_index(drop=True)
        return filtered_train_df



def automl_filter_flip(train_df, dataset_dir):
    """An augmentation strategy: permute the training set after filtering by AutoML model."""
    filtered_train_df = automl_filter(train_df, dataset_dir)
    dataset_name = dataset_dir.split('/')[-1]
    # swap the left and right records
    flipped_ftd = filtered_train_df.copy()
    left_columns = [col for col in filtered_train_df.columns if col.endswith('_l')]
    right_columns = [col for col in filtered_train_df.columns if col.endswith('_r')]
    rename_schema = dict(zip(left_columns+right_columns, right_columns+left_columns))
    flipped_ftd = flipped_ftd.rename(columns=rename_schema)
    new_train_df = pd.concat([filtered_train_df, flipped_ftd.sample(frac=0.1, random_state=42)]).reset_index(drop=True)
    return new_train_df


def read_single_row_data(dataset_dir, mode, sample_func='', print_info=True, seed=42):
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(dataset_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))

    if sample_func:
        sample_func = eval(sample_func)
        train_df = sample_func(train_df, dataset_dir)

    train_df = df_serializer(train_df, mode, seed)
    valid_df = df_serializer(valid_df, mode, seed)
    test_df = df_serializer(test_df, mode, seed)

    if print_info:
        dataset_name = dataset_dir.split('/')[-1]
        print(f"We will use {mode} to serialize the {dataset_name} dataset.", flush=True)
        print(f"Examples(row level) after the serialization is:{test_df.iloc[0]['text']}", flush=True)

    return train_df, valid_df, test_df


def read_multi_row_data(dataset_dirs, mode='mode1', sample_func='one_pos_two_neg', print_info=True, seed=42):
    dfs = [read_single_row_data(dataset_dir, mode, sample_func, print_info=False, seed=seed) for dataset_dir in dataset_dirs]
    train_dfs, valid_dfs, test_dfs = zip(*dfs)

    if print_info:
        print(f"We will use {mode} to serialize all datasets.", flush=True)
        print(f"The {sample_func} function is applied to conduct sampling/augmentation for all datasets.", flush=True)
        sample_texts = [valid_df.iloc[0]['text'] for valid_df in valid_dfs]
        print(f"Examples(row level) after the serialization are:", flush=True)
        [print(sample_text, flush=True) for sample_text in sample_texts]

    concat_train_df = pd.concat(train_dfs, ignore_index=True)
    concat_valid_df = pd.concat(valid_dfs, ignore_index=True)
    concat_valid_df = concat_valid_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    concat_test_df = pd.concat(test_dfs, ignore_index=True)

    print(f'The size of the row level concatenation for training and validation are: {len(concat_train_df)}, '
          f'{len(concat_valid_df)}', flush=True)

    return concat_train_df, concat_valid_df, concat_test_df


def downsample_attr_pairs(group):
    group_pos_num = group['label'].sum()
    group_neg_num = min(len(group) - group_pos_num, 2)

    group_pos_partition = group[group['label'] == 1]
    if group_neg_num > 2 * group_pos_num:
        group_neg_partition = group[group['label'] == 0].sample(n=2 * group_pos_num, random_state=1)
    else:
        group_neg_partition = group[group['label'] == 0]

    sampled_group = pd.concat([group_pos_partition, group_neg_partition])

    if len(sampled_group) > 500:
        sampled_group = sampled_group.sample(n=500, random_state=1)

    return sampled_group


def read_multi_attr_data(dataset_dirs, mode='mode1'):
    train_dfs = [pd.read_csv(os.path.join(dataset_dir, 'attr_train.csv')) for dataset_dir in dataset_dirs]
    valid_dfs = [pd.read_csv(os.path.join(dataset_dir, 'attr_valid.csv')) for dataset_dir in dataset_dirs]
    test_dfs = [pd.read_csv(os.path.join(dataset_dir, 'attr_test.csv')) for dataset_dir in dataset_dirs]

    concat_train_df = pd.concat(train_dfs, ignore_index=True)
    concat_valid_df = pd.concat(valid_dfs, ignore_index=True)
    concat_test_df = pd.concat(test_dfs, ignore_index=True)
    final_train_df = concat_train_df.groupby('attribute').apply(downsample_attr_pairs).reset_index(drop=True)[
        ['left_value', 'right_value', 'label']]
    final_train_df.columns = ['value_l', 'value_r', 'label']
    final_valid_df = concat_valid_df.groupby('attribute').apply(downsample_attr_pairs).reset_index(drop=True)[
        ['left_value', 'right_value', 'label']]
    final_valid_df.columns = ['value_l', 'value_r', 'label']
    final_test_df = concat_test_df.groupby('attribute').apply(downsample_attr_pairs).reset_index(drop=True)[
        ['left_value', 'right_value', 'label']]
    final_test_df.columns = ['value_l', 'value_r', 'label']
    final_train_df = df_serializer(final_train_df, mode)
    final_valid_df = df_serializer(final_valid_df, mode)
    final_test_df = df_serializer(final_test_df, mode)

    return final_train_df, final_valid_df, final_test_df
