def get_model(model_name, args):
    name = model_name.lower()
    if name == "gmm_ensemble":
        from models.gmm_ensemble import Learner
    else:
        assert 0

    return Learner(args)