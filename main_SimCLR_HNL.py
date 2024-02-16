import argparse
from pytorch_lightning import Trainer, seed_everything
from models.mlp import UnimodalLinearEvaluator

from utils.utils_exp import generate_experiment_id, load_yaml_to_dict
from utils.utils_train import *
from models.simclr_hn import SimCLRUnimodal as SimCLRUnimodal_HNL
from models.simclr import SimCLRUnimodal

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # configs paths
    parser.add_argument('--experiment_config_path', required=True)
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')
    parser.add_argument('--augmentations_path', default='configs/augmentations.yaml')
    parser.add_argument('--tuning_config_path', default=None)

    # data and models
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--protocol', default='cross_subject')
    parser.add_argument('--framework', default='simclr')
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_save_path', default='./model_weights')

    # used to run only in fine tuning mode
    parser.add_argument('--fine_tuning', action='store_true')
    parser.add_argument('--fine_tuning_ckpt_path', help='Path to a pretrained encoder. Required if running with --fine_tuning.')

    # other training configs
    parser.add_argument('--no_ckpt', action='store_true', default=False)
    parser.add_argument('--online-eval', action='store_true', default=False)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--sweep', action='store_true', default=False, help='Set automatically if running in WandB sweep mode. You do not need to set this manually.')

    # hyperparameter
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_epochs_ssl', default=300, type=int)
    parser.add_argument('--num_epochs_fine', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--tau_plus', default=0.1, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--estimator', default='easy')
    parser.add_argument('--ssl_lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--weight_decay_fine', default=0, type=float)
    
    return parser.parse_args()


def ssl_pre_training(args, modality, cfg, dataset_cfg, experiment_id, loggers_list, loggers_dict):
    seed_everything(args.seed)
    num_epochs = args.num_epochs_ssl
    augmentations_dict = load_yaml_to_dict(args.augmentations_path)
    flat_augmentations_dict = nested_to_flat_dict({"augmentations": augmentations_dict}) # need flat structure for wandb sweep to properly overwrite it

    # if using wandb and performing a sweep, overwrite the config params with the sweep params.
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment

        # Take some specific parameters.
        num_epochs = _wandb.config["num_epochs_ssl"]
        
        # Take SSL model kwargs and merge with experiment config.
        ssl_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('ssl.')}
        ssl_kwargs_dict = flat_to_nested_dict(ssl_key_values)
        if ssl_kwargs_dict != {}:
            cfg['modalities'][modality]['model']['ssl']['kwargs'] = {**cfg['modalities'][modality]['model']['ssl']['kwargs'], **ssl_kwargs_dict['ssl']}

        # Take encoder kwargs and merge with experiment config.
        encoder_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('encoder.')}
        encoder_kwargs_dict = flat_to_nested_dict(encoder_key_values)
        if encoder_kwargs_dict != {}:
            cfg['modalities'][modality]['model'][args.model]['kwargs'] = {**cfg['modalities'][modality]['model'][args.model]['kwargs'], **encoder_kwargs_dict['encoder']}

        # Take augmentation config from sweep and merge with default config.
        augmentation_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('augmentations.')}
        flat_augmentations_dict = {**flat_augmentations_dict, **augmentation_key_values}
        augmentations_dict = flat_to_nested_dict(flat_augmentations_dict)['augmentations']

    # take sample_length cfg from model definition and overwrite transform args
    model_cfg = cfg['modalities'][modality]['model'][args.model]
    transform_cfg = cfg['modalities'][modality]['transforms']
    model_cfg, transform_cfg = check_sampling_cfg(model_cfg, transform_cfg)

    # initialize transforms: modailty transforms + random transformations for view generation
    train_transforms, test_transforms = init_transforms(modality, transform_cfg, ssl_random_augmentations=True, random_augmentations_dict=augmentations_dict)
    
    # init datamodule with ssl flag
    batch_size = args.batch_size
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=[modality], batch_size=batch_size,
        split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
        ssl=True, n_views=2, num_workers=args.num_workers)

    # Merge general model params with dataset-specific model params.
    model_cfg['kwargs'] = {**dataset_cfg[modality], **model_cfg['kwargs']}
    
    # initialize encoder and unimodal ssl framework model
    encoder = init_ssl_encoder(model_cfg)
    if args.framework == 'simclr':
        model = SimCLRUnimodal(modality, encoder, encoder.out_size, args.temperature, batch_size, args.ssl_lr, args.weight_decay, **cfg['modalities'][modality]['model']['ssl']['kwargs'])
    elif args.framework == 'simclr_hnl':
        model = SimCLRUnimodal_HNL(modality, encoder, encoder.out_size, args.temperature, args.tau_plus, args.beta, args.estimator, batch_size, args.ssl_lr, args.momentum, args.weight_decay, **cfg['modalities'][modality]['model']['ssl']['kwargs'])


    callbacks = setup_callbacks_ssl(
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        dataset               = args.dataset, 
        model                 = args.model, 
        experiment_id         = experiment_id,
    )

    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='logs', 
        val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0, callbacks=callbacks)

    trainer.fit(model, datamodule)
    return model.encoder, cfg


def fine_tuning(args, modality, cfg, dataset_cfg, encoder, loggers_list, loggers_dict, experiment_id, limited_k=None):
    seed_everything(args.seed) # reset seed for consistency in results
    batch_size = cfg['experiment']['batch_size_fine_tuning']
    num_epochs = args.num_epochs_fine

    # if using wandb and performing a sweep, overwrite some config params with the sweep params.
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment
        batch_size = _wandb.config["batch_size_fine_tuning"]
        num_epochs = _wandb.config["num_epochs_fine_tuning"]
        
    model = UnimodalLinearEvaluator(modality, encoder, encoder.out_size, dataset_cfg["n_classes"], args.weight_decay_fine)

    callbacks = setup_callbacks(
        early_stopping_metric = "val_loss",
        early_stopping_mode   = "min",
        class_names           = dataset_cfg["class_names"],
        num_classes           = len(dataset_cfg["class_names"]),
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        metric                = 'val_' + dataset_cfg['main_metric'], 
        dataset               = args.dataset, 
        model                 = 'ssl_finetuned_' + args.framework + '_' + args.model, 
        experiment_id         = experiment_id
    )

    train_transforms, test_transforms = init_transforms(modality, cfg['modalities'][modality]['transforms'])
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=[modality], batch_size=batch_size,
        split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
        num_workers=args.num_workers, limited_k=limited_k)

    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='logs', 
        val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0, callbacks=callbacks)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}

    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics


def init_loggers(args, modality, cfg, experiment_id, fine_tune_only=False):
    experiment_info = { # default values; may be overrided by sweep config
        "dataset": args.dataset,
        "model": cfg['modalities'][modality]['model'][args.model]['encoder_class_name'],
        "seed": args.seed,
        "temperature": args.temperature,
        "tau_plus": args.tau_plus,
        "beta": args.beta,
        "estimator": args.estimator,
        "ssl_lr": args.ssl_lr,
        "momentum":args.momentum,
        "weight_decay": args.weight_decay,
        "weight_decay_fine": args.weight_decay_fine
    }
    if not fine_tune_only:
        num_epochs = args.num_epochs_ssl
        augmentations_dict = load_yaml_to_dict(args.augmentations_path)
        flat_augmentations_dict = nested_to_flat_dict({"augmentations": augmentations_dict}) # need flat structure for wandb sweep to properly overwrite it
        additional_info = { # default values; may be overrided by sweep config
            "ssl_framework": args.framework,
            "num_epochs_ssl": num_epochs,
            "num_epochs_fine_tuning": args.num_epochs_fine,
            "batch_size_fine_tuning": cfg['experiment']['batch_size_fine_tuning'],
            **flat_augmentations_dict,
        }
        experiment_info = {**experiment_info, **additional_info}
    
    loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, modality=modality, dataset=args.dataset, 
        experiment_id=experiment_id, experiment_config_path=args.experiment_config_path, approach='ssl')
    return loggers_list, loggers_dict

def run_one_experiment(args, cfg, dataset_cfg):
    experiment_id = generate_experiment_id()
    modality = list(cfg['modalities'].keys())[0]
    loggers_list, loggers_dict = init_loggers(args, modality, cfg, experiment_id, fine_tune_only=False)

    encoder, cfg = ssl_pre_training(args, modality, cfg, dataset_cfg, experiment_id, loggers_list, loggers_dict)
    result_metrics = fine_tuning(args, modality, cfg, dataset_cfg, encoder, loggers_list, loggers_dict, experiment_id)
    return result_metrics

def run_fine_tuning_only(args, cfg, dataset_cfg):
    experiment_id = generate_experiment_id()
    modality = list(cfg['modalities'].keys())[0]
    loggers_list, loggers_dict = init_loggers(args, modality, cfg, experiment_id, fine_tune_only=False)

    model_cfg = cfg['modalities'][modality]['model'][args.model]
    model_cfg['kwargs'] = {**dataset_cfg[modality], **model_cfg['kwargs']}
    pre_trained_model = init_ssl_pretrained(model_cfg, args.fine_tuning_ckpt_path)
    encoder = getattr(pre_trained_model, 'encoder')
    fine_tuning(args, modality, cfg, dataset_cfg, encoder, loggers_list, loggers_dict, experiment_id)

def validate_args(args):
    if args.fine_tuning and not args.fine_tuning_ckpt_path:
        print("Need to provide --fine_tuning_ckpt_path if running with --fine_tuning!")
        exit(1)

def main():
    args = parse_arguments()
    validate_args(args)
    cfg = load_yaml_to_dict(args.experiment_config_path)
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]
    
    if args.fine_tuning:
        run_fine_tuning_only(args, cfg, dataset_cfg)
    else:
        run_one_experiment(args, cfg, dataset_cfg)

if __name__ == '__main__':
    main()