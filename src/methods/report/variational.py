from src.methods.report.base import BaseReport
class VarBaseReport(BaseReport): 
    def __call__(self, callback: dict) -> None : 

        print(f'Epoch [{callback["num_epoch"]}/{callback["total_num_epoch"]}],'\
               f'Loss:{callback["total_loss"]}, KL Loss: {callback["kl_loss"]}. FitLoss: {callback["fit_loss"]},Accuracy:{callback["accuracy"]},'\
               f'Validation Loss:{callback["val_total_loss"]},Validation Accuracy:{callback["val_accuracy"]}, Prune parameters: {callback["cnt_prune_parameters"]}/{callback["cnt_params"]},'\
                f'Beta: {callback["beta"]}')

