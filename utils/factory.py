def get_model(model_name, args):
    name = model_name.lower()
    if name in ["alldata_statistics", "partdata_statistics_cov"]:
        from models.test_statistics import Learner
    elif name == "joint_training":
        from models.joint_training import Learner
    elif name == "alldata_statistics_imbalance":
        from models.test_statistics_imbalace import Learner
    elif name == "multi_proxy":
        from models.multi_proxy import Learner
    elif name == "mahala_distance":
        from models.mahala_distance import Learner
    elif name == "ensemble":
        from models.ensemble import Learner
    elif name == "gmm":
        from models.gmm import Learner
    elif name == "gmm_ensemble":
        from models.gmm_ensemble import Learner
    elif name == "visual_classifier":
        from models.visual_classifier import Learner
    elif name == "tsne":
        from models.tsnefeature import Learner
    elif name == "adaptive_lambda":
        from models.adaptive_lambda import Learner
    else:
        assert 0

    return Learner(args)