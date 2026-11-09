from .MSACT import MCSA_Transformer, MCSA_AcousticVibration

# 工业部署精简版模型字典，仅保留本项目核心 Backbone
model_dict = {
    'MCSAT': MCSA_Transformer,
    'MCSAT_AV': MCSA_AcousticVibration,
}
