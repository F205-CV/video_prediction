
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'seperate_train':
        #assert(opt.train_mode == 'seperate')
        from .seperate_model import SeperateModel
        model = SeperateModel()
    elif opt.model == 'joint_train':
        #assert(opt.train_mode == 'joint')
        from .joint_model import JointModel
        model = JointModel()
    elif opt.model == 'test':
        #assert(opt.test_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
