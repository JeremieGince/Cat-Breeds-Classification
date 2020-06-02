if __name__ == '__main__':
    from trainer import Trainer
    from util import plotHistory, plot_cat_colorization_prediction_samples
    from models import CatBreedsClassifier, CatColorizer
    from Dataset import CatBreedsClassifierDataset, CatColorizerOverfittingDataset
    from hyperparameters import *

    BATCH_SIZE = 1
    FEATURES_TRAINING_EPOCHS = 1_000
    GAMUT_SIZE = 5

    col_overfitting_dataset = CatColorizerOverfittingDataset(
        gamut_size=GAMUT_SIZE,
        bins=BINS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    col_overfitting_dataset.show_gamut_probabilities(savefig=False)
    col_overfitting_dataset.show_gamut_probabilities(rebin=True, savefig=False)

    col_model_manager = CatColorizer(
        *col_overfitting_dataset.get_gamut_params(),
        fusion_depth=FUSION_DEPTH,
        img_size=col_overfitting_dataset.IMG_SIZE,
        name=f"CatColorizer_overfitted_gamut-{col_overfitting_dataset.GAMUT_SIZE}",
    )
    col_model_manager.build_and_compile()
    print(col_model_manager.summary)

    # Training the features
    col_model_manager.load()
    col_trainer = Trainer(
        col_model_manager,
        col_overfitting_dataset,
        use_saving_callback=True,
        load_on_start=True,
        network_callback_args={
            "verbose": True,
            "save_freq": 100
        }
    )
    col_trainer.train(FEATURES_TRAINING_EPOCHS)
    plotHistory(col_model_manager.history, savename="overfitted_training_curve")
    plot_cat_colorization_prediction_samples(col_model_manager,
                                             savename="overfitted_cat_colorization_prediction_samples")
