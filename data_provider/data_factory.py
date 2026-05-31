from data_provider.data_loader import Dataset_Custom_vlbm, RobustTimeSeriesDatasetForCSV_vlbm, RobustTimeSeriesDatasetForNPY0_vlbm, \
    RobustTimeSeriesDatasetForNPY1_vlbm, RobustTimeSeriesDatasetForNPY_vlbm,RobustTimeSeriesDatasetForNPY_vlbm_allow, \
    Dataset_PEMS_vlbm_ConsistentWithV1
    
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'vlbm' : Dataset_Custom_vlbm,
    'csv_by_ood_vlbm' :RobustTimeSeriesDatasetForCSV_vlbm,
    'vlbm_pems_0': RobustTimeSeriesDatasetForNPY0_vlbm, 
    'vlbm_pems_1': RobustTimeSeriesDatasetForNPY1_vlbm,
    'pems_vlbm_v1' : Dataset_PEMS_vlbm_ConsistentWithV1,
    'npy_by_vlbm': RobustTimeSeriesDatasetForNPY_vlbm,
    'npy_by_vlbm_allow': RobustTimeSeriesDatasetForNPY_vlbm_allow,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'long_term_forecast_vlbm':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            interval = args.interval
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            pin_memory=True, persistent_workers=True,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'ood_vlbm':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            label_path=args.label_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            interval = args.interval
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            pin_memory=True, persistent_workers=True,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'ood_vlbm_large':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            label_path=args.label_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            interval = args.interval
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            pin_memory=True, persistent_workers=True,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            pin_memory=True, persistent_workers=True,
            drop_last=drop_last)
        return data_set, data_loader
